"""Small classifier contracts; imports tensor libraries only during explicit execution."""
import math

ARCHITECTURE = 'mobilenet-v3-large'
WEIGHTS_FILENAME = 'mobilenet_v3_large-5c1a4163.pth'
WEIGHTS_URL = 'https://download.pytorch.org/models/' + WEIGHTS_FILENAME
WEIGHTS_SHA256 = '5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997'


def validate_recipe(config):
    if config['resolution'] != 224 or not 0 <= config['context'] <= .16:
        raise ValueError('Reviewed student uses complete224 crops with context0..0.16')
    if not 0 <= config['distillationAlpha'] < 1 or not 1 <= config['temperature'] <= 10:
        raise ValueError('Invalid distillation recipe')
    if not 0 <= config['teacherMinimumConfidence'] <= 1:
        raise ValueError('Invalid teacher confidence gate')
    if min(config['epochs'], config['batch'], config['gradientAccumulation']) < 1 or config['workers'] < 0:
        raise ValueError('Invalid epoch/batch/worker recipe')
    if not all(math.isfinite(v) for v in (config['lr'], config['maximumTrainMinutes'], config['distillationAlpha'], config['temperature'])) or not 0 < config['lr'] < .1 or config['maximumTrainMinutes'] <= 0:
        raise ValueError('Finite positive learning rate and budget required')
    return config


def build_student(pretrained=None):
    from torch import nn
    from torchvision.models import mobilenet_v3_large
    model = mobilenet_v3_large(weights=None)
    if pretrained is not None:
        import torch
        model.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True), strict=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 42)
    return model


def distillation_loss(logits, labels, teacher_logits, class_weights, config):
    """Disagreeing/uncertain teacher never suppresses the ground-truth CE target."""
    import torch
    from torch.nn import functional as F
    hard = F.cross_entropy(logits.float(), labels, weight=class_weights, label_smoothing=.02, reduction='none')
    if teacher_logits is None or config['distillationAlpha'] == 0:
        return hard.mean(), hard.mean().detach(), logits.new_zeros(()), logits.new_zeros(())
    probabilities = teacher_logits.float().detach().softmax(-1)
    confidence, predicted = probabilities.max(-1)
    eligible = ((predicted == labels) & (confidence >= config['teacherMinimumConfidence'])).float()
    t = config['temperature']
    soft = F.kl_div(F.log_softmax(logits.float()/t, -1), F.softmax(teacher_logits.float().detach()/t, -1), reduction='none').sum(-1)*(t*t)
    alpha = config['distillationAlpha']*eligible
    return ((1-alpha)*hard + alpha*soft).mean(), hard.mean().detach(), (soft*eligible).mean().detach(), eligible.mean().detach()


def calibration_indices(records, limit=672, seed=2026):
    """Deterministic train-only round robin, preferring distinct source groups per class."""
    import hashlib
    from collections import defaultdict
    if limit < 42:
        raise ValueError('Calibration must cover all42 classes')
    by_class = defaultdict(list)
    for index, row in enumerate(records):
        label = int(row['label'])
        if label not in range(42):
            raise ValueError('Unexpected calibration class')
        key = hashlib.sha256(f"{seed}:{row['annotationId']}:{row['groupId']}".encode()).hexdigest()
        by_class[label].append((key, index))
    if set(by_class) != set(range(42)):
        raise ValueError('Calibration requires train examples of every class')
    for label, values in by_class.items():
        values.sort()
        unique, repeated, groups = [], [], set()
        for key, index in values:
            group = records[index]['groupId']
            (repeated if group in groups else unique).append(index)
            groups.add(group)
        by_class[label] = unique + repeated
    selected, position = [], 0
    while len(selected) < min(limit, len(records)):
        added = False
        for label in range(42):
            if position < len(by_class[label]) and len(selected) < limit:
                selected.append(by_class[label][position]); added = True
        if not added: break
        position += 1
    return selected
