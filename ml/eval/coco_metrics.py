"""COCO AP with one-based evaluator-local annotation IDs; source bytes stay frozen."""


def coco_metrics(images, annotations, predictions, classes):
    if not predictions or not annotations:
        return {"mAP50_95":0.,"mAP50":0.,"mAP75":0.,"maxDetections":100}
    from faster_coco_eval import COCO, COCOeval_faster
    # COCO match arrays reserve zero for unmatched. Source v4 annotation IDs are
    # zero-based, so only the temporary evaluator representation is renumbered.
    normalized=[dict(annotation,id=index+1) for index,annotation in enumerate(annotations)]
    truth=COCO({'images':images,'annotations':normalized,
        'categories':[{'id':int(index)+1,'name':token} for index,token in classes.items()]})
    detected=truth.loadRes(predictions)
    evaluator=COCOeval_faster(truth,detected,'bbox',print_function=lambda *args,**kwargs:None)
    evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    return {'mAP50_95':float(evaluator.stats[0]),'mAP50':float(evaluator.stats[1]),
        'mAP75':float(evaluator.stats[2]),'maxDetections':100,'annotationIdPolicy':'one_based_evaluator_local_copy'}
