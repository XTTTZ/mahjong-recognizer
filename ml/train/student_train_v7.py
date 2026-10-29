"""Bounded supervised/KD MobileNetV3 training; --preflight-only never imports torch."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
import time
from ml.data.prepare_crops import verify_prepared
from ml.data.prepare_crop_cache_v6 import verify_cache
from ml.inference.cascade_model import build_classifier, image_tensor
from ml.train.provenance import atomic_json, dataset_contract, sha256
from ml.train.student_common import ARCHITECTURE, WEIGHTS_FILENAME, WEIGHTS_SHA256, build_student, distillation_loss, validate_recipe
from ml.train.v6_common import parent_contract, completed_epoch_budget
from ml.train.v6_augment import cached_crop_face
from ml.train.yolo_train import admit_output, checkpoint_receipt, shared_signature


class PairedCrops:
    def __init__(self, cache, split, config, teacher_resolution=320, teacher_context=.08):
        self.root, self.split, self.config = Path(cache), split, config
        self.records = json.loads((self.root/f'{split}.json').read_text())['records']
        self.teacher_resolution, self.teacher_context = teacher_resolution, teacher_context

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        from ml.train.rtdetr_train import photometric
        row = self.records[index]; path = (self.root/row['cacheFile']).resolve()
        if not path.is_relative_to(self.root.resolve()): raise ValueError('Cache path escape')
        training = self.split == 'train'
        # Both views repeat the exact global crop jitter, photometric factors and
        # quarter-turn. Different target resolutions do not alter RNG decisions.
        seed = random.getrandbits(63) if training else 0
        def face(source, resolution, context):
            rng = random.Random(seed)
            result = cached_crop_face(source, row, resolution, context, rng if training else None)
            if training:
                result = photometric(result, rng)
                result = result.rotate(rng.choice((0, 90, 180, 270)))
            return image_tensor(result)
        with Image.open(path) as source:
            student = face(source, self.config['resolution'], self.config['context'])
            # Teacher is only used for train, never validation pseudo-targets.
            teacher = face(source, self.teacher_resolution, self.teacher_context) if training and self.config['distillationAlpha'] else student
        return student, int(row['label']), teacher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'crops', 'cache', 'weights', 'teacher-run', 'teacher-checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--teacher-terminal-receipt', type=Path)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch', type=int, default=16)
    p.add_argument('--gradient-accumulation', type=int, default=2); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--resolution', type=int, choices=(224,), default=224); p.add_argument('--context', type=float, default=.08)
    p.add_argument('--lr', type=float, default=.0003); p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--distillation-alpha', type=float, default=0); p.add_argument('--temperature', type=float, default=2.)
    p.add_argument('--teacher-min-confidence', type=float, default=.7)
    p.add_argument('--max-train-minutes', type=float, default=40); p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    config = validate_recipe({'epochs':args.epochs, 'batch':args.batch, 'gradientAccumulation':args.gradient_accumulation,
        'workers':args.workers, 'resolution':args.resolution, 'context':args.context, 'lr':args.lr, 'seed':args.seed,
        'distillationAlpha':args.distillation_alpha, 'temperature':args.temperature,
        'teacherMinimumConfidence':args.teacher_min_confidence, 'maximumTrainMinutes':args.max_train_minutes,
        'augmentation':'photometric_global_jitter_and_quarter_turn', 'labelSmoothing':.02,
        'teacherGate':'correct_gt_class_and_minimum_untempered_probability', 'inputMode':'complete_crop_letterbox_imagenet_normalization'})
    contract = dataset_contract(args.dataset, verify_images=True)
    if contract['ontology'] != 'faces42': raise ValueError('Student requires42 reviewed faces')
    ready = (args.crops/'prepared.json').is_file() and (args.cache/'prepared.json').is_file()
    if ready:
        verify_prepared(args.crops, contract); verify_cache(args.cache, contract, args.crops)
        labels={int(row['label']) for row in json.loads((args.cache/'train.json').read_text())['records']}
        if labels!=set(range(42)):raise ValueError('Every class must be represented in the prepared train cache')
    teacher_run, teacher_evidence = parent_contract(args.teacher_run, args.teacher_checkpoint, contract, args.teacher_terminal_receipt, generic=False)
    if teacher_run['architecture'] != 'efficientnet-v2-s': raise ValueError('Use the frozen cascade classifier teacher')
    if args.weights.name != WEIGHTS_FILENAME or not args.weights.is_file(): raise FileNotFoundError('Cache the official MobileNetV3-Large V2 weights')
    if sha256(args.weights)!=WEIGHTS_SHA256:raise ValueError('Official student pretrained weights SHA256 mismatch')
    signature = shared_signature(contract, config, args.weights)
    signature.update(teacher=teacher_evidence, derivativeSha256=sha256(args.crops/'prepared.json') if ready else None,
                     cacheSha256=sha256(args.cache/'prepared.json') if ready else None)
    root = Path(__file__).resolve().parents[1]
    signature['code'].update({name:sha256(root/name) for name in ('train/student_train_v7.py','train/student_common.py','inference/student_model.py','train/v6_common.py','train/v6_augment.py','train/rtdetr_train.py','data/prepare_crop_cache_v6.py','data/prepare_crops.py',
        'eval/student_evaluate_v7.py','train/student_quantize_v7.py','eval/recover_terminal.py','eval/evaluate_yolo_candidates.py','eval/yolo_evaluate.py','eval/coco_metrics.py')})
    if args.preflight_only:
        print(json.dumps({'trainingStarted':False,'derivativeReady':ready,'config':config,'teacher':teacher_evidence,'weightsSha256':signature['pretrainedSha256'],'dataContract':contract})); return
    if not ready: raise FileNotFoundError('Prepare verified lossless crop cache first')
    if (args.output/'run.json').exists(): raise FileExistsError('Use a new student run')
    if importlib.metadata.version('torch').split('+')[0] != '2.8.0' or importlib.metadata.version('torchvision').split('+')[0] != '0.23.0':
        raise RuntimeError('Use pinned Torch2.8/Torchvision0.23 environment')
    import torch
    from torch.utils.data import DataLoader
    from ml.train.rtdetr_train import seed_worker
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    admit_output(args.output)
    run = {'architecture':ARCHITECTURE,'classes':contract['classes'],'ontology':'faces42','resolution':224,'config':config,
        'resumeSignature':signature,'startedAt':datetime.now(timezone.utc).isoformat(),'testOpened':False,
        'annotationSha256':contract['splits']['train']['annotationSha256'], 'validationAnnotationSha256':contract['splits']['valid']['annotationSha256'],
        'validationScope':'oracle_truth_crops_diagnostic_only','teacher':teacher_evidence,'plannedEpochs':args.epochs}
    atomic_json(args.output/'run.json', run)
    student = build_student(args.weights).cuda(); teacher = None
    if args.distillation_alpha:
        teacher = build_classifier()
        saved = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=True)
        expected = hashlib.sha256(json.dumps(teacher_run['resumeSignature'],sort_keys=True).encode()).hexdigest()
        if saved['classes'] != contract['classes'] or saved['signatureSha256'] != expected: raise ValueError('Teacher payload differs from signed source')
        teacher.load_state_dict(saved['model'],strict=True); teacher.cuda().eval(); teacher.requires_grad_(False)
    train = PairedCrops(args.cache,'train',config,teacher_run['config']['resolution'],teacher_run['config']['context'])
    valid = PairedCrops(args.cache,'valid',config)
    counts = Counter(row['label'] for row in train.records)
    if set(counts) != set(range(42)): raise ValueError('Every class must be represented')
    weights = torch.tensor([min(4.,(len(train)/(42*counts[i]))**.5) for i in range(42)], device='cuda'); weights /= weights.mean()
    generator = torch.Generator().manual_seed(args.seed)
    training = DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,worker_init_fn=seed_worker,generator=generator)
    validation = DataLoader(valid,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True,worker_init_fn=seed_worker)
    optimizer = torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.05)
    best = -1.; completed = 0; completion = 'epoch_limit_reached'; started = time.monotonic()
    with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=['epoch','train/loss','train/ce','train/kd','teacher_eligible_fraction','valid/loss','oracle_crop_top1','learningRate']); writer.writeheader()
        for epoch in range(args.epochs):
            epoch_started = time.monotonic(); student.train(); optimizer.zero_grad(set_to_none=True)
            totals = [0.,0.,0.,0.]; seen = 0
            for step,(images,labels,teacher_images) in enumerate(training):
                images,labels = images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): teacher_logits = teacher(teacher_images.cuda(non_blocking=True))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits = student(images); loss,hard,soft,eligible = distillation_loss(logits,labels,teacher_logits,weights,config)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite student loss; independent queue continues')
                tail = len(training)%args.gradient_accumulation
                denominator = tail if tail and step >= len(training)-tail else args.gradient_accumulation
                (loss/denominator).backward()
                if (step+1)%args.gradient_accumulation == 0 or step+1 == len(training):
                    torch.nn.utils.clip_grad_norm_(student.parameters(),1.); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                for i,value in enumerate((loss.detach(),hard,soft,eligible)): totals[i] += float(value)*len(labels)
                seen += len(labels)
            student.eval(); correct = count = 0; valid_loss = 0.
            with torch.inference_mode():
                for images,labels,_ in validation:
                    images,labels = images.cuda(),labels.cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16): logits=student(images)
                    if not torch.isfinite(logits).all(): raise RuntimeError('Non-finite student validation')
                    correct += int((logits.argmax(-1)==labels).sum()); count += len(labels)
                    valid_loss += float(torch.nn.functional.cross_entropy(logits.float(),labels))*len(labels)
            accuracy = correct/count
            row = dict(zip(('train/loss','train/ce','train/kd','teacher_eligible_fraction'),(value/seen for value in totals)))
            row.update(epoch=epoch,**{'valid/loss':valid_loss/count,'oracle_crop_top1':accuracy,'learningRate':optimizer.param_groups[0]['lr']})
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True);scheduler.step()
            saved = {'architecture':ARCHITECTURE,'model':student.state_dict(),'classes':contract['classes'],'config':config,'epoch':epoch,
                'signatureSha256':hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'pythonRngState':random.getstate(),
                'torchRngState':torch.get_rng_state(),'cudaRngStates':torch.cuda.get_rng_state_all(),'loaderRngState':generator.get_state()}
            for name in (('last.pt','best.pt') if accuracy > best else ('last.pt',)):
                temporary=args.output/(name+'.tmp');torch.save(saved,temporary);temporary.replace(args.output/name)
                checkpoint_receipt(args.output/name,signature,contract['classes'])
            best=max(best,accuracy);completed=epoch+1
            reason=completed_epoch_budget(completed,args.epochs,time.monotonic()-started,args.max_train_minutes,time.monotonic()-epoch_started)
            atomic_json(args.output/'epoch-completion.json',{'completedEpochs':completed,'checkpointSavedAndValidated':True,'stopReason':reason})
            if reason:completion=reason;break
    run.update(finishedAt=datetime.now(timezone.utc).isoformat(),completedEpochs=completed,completionReason=completion,
        elapsedTrainSeconds=time.monotonic()-started,bestOracleCropTop1=best,convergenceClaim=False,
        note='Classifier-stage distillation only; detector remains the frozen V26 cascade component. Crop accuracy is diagnostic, not whole-hand accuracy.')
    atomic_json(args.output/'run.json',run)


if __name__ == '__main__': main()
