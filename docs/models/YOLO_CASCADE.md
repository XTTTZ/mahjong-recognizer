# YOLO 与定位后分类实验

2026-10-26准备；没有启动这些模型的训练。YOLO、classifier的CPU数据合同和后处理已测试；GPU收敛、显存、精度、Linux导出和耗时仍需要提交后的真实实验确认。

## 比较什么

| 路线 | 目的 | 输入与预处理 | 关键区别 |
| --- | --- | --- | --- |
| YOLOv8s直接42类检测 | 建立与旧方案可解释的检测基线 | 全图960，Ultralytics正方形letterbox | 保留v8s作为旧模型规模对照，非预设最终选择 |
| YOLO26m直接42类检测 | 比较现代检测器和较大容量 | 全图960，同样的split | 官方YOLO26支持原生端到端检测；使用发布权重对应原生路径，固定包版本 |
| YOLO26s单类定位→EfficientNetV2-S | 把“找牌”和“细分纹理”拆开 | 定位960；实际预测框加8%上下文，完整保留牌面比例，letterbox到320 | 分类细节无需受全图缩放后小目标的限制；定位漏牌/误框仍会导致整手失败 |

RF-DETR与RT-DETR仍是独立候选。模型更大、新版本或二阶段本身都不是准确率结论。第一轮先比较完整手牌结果，再决定是否扩大epoch、数据增强、分类器容量或做集成。

## 数据与评价纪律

全部新路线使用冻结的v4 42类牌面映射。牌背作为背景；同原图的预增强与相近相机burst不会跨train/valid。这里的2629张训练图含同源预增强，不能按2629次独立采集理解。原test保持封存。

`prepare_yolo`把train/valid转为YOLO标签，复用原图字节，不重新编码；generic版本只把所有牌面合为类别0。每份衍生数据保存源COCO/图片树、标签、图片和准备代码的哈希。训练前重新校验图片、标签与映射。

`prepare_crops`保存包含原图、框、类别和source group的索引，训练时按索引即时裁图，避免把同一张图的牌拆散后随机分组。train图的框用于classifier训练；valid真值框只能给`oracle_crop_top1`诊断分类能力。最终`cascade_evaluate`一定从完整valid照片开始，使用定位器预测框，再分类、融合分数和统一后处理。漏检、多检、定位错误、分类错误、模型异常均计入整手分母，不通过补齐14张或强制4张上限修正预测。

报告并列给出原生≥10张牌照片的整手全部正确率、模型全部正确率、接受覆盖率、接受结果准确率和COCO AP@100。API与评估共用严格`score > threshold`、bbox裁剪、200张上限、低置信拒绝策略。cascade分数为检测分数乘分类softmax，尚未校准；仅在valid选择工作点，不能将其当真实正确概率。报告区分oracle ROI、单牌和完整原图。现有valid仍不是独立棋牌室手机测试集。

## 训练策略

YOLO使用本地官方COCO预训练权重，固定Ultralytics8.4.189。AdamW、较小学习率、cosine、warmup3、batch按12GB显存保守设置，按实际batch积累到nominal32。为了保留麻将字形，不翻转、不mosaic/mixup、不随机裁切；photo版本只做轻量HSV变化。generic定位器与42类检测器使用独立头、独立run与checkpoint，不混用恢复。

EfficientNetV2-S使用官方ImageNet预训练，从本地加载并替换42类头；完整裁图保持比例，使用ImageNet归一化；仅train有轻量亮度/对比/颜色/模糊/JPEG增强和小幅框抖动，抵抗推理框与真值框差距。classifier使用上限4倍的逆平方根类别加权loss、label smoothing0.02和AdamW/cosine。320像素建议先用batch16、gradient accumulation2。没有执行GPU试训，因此这是待测配置，OOM由队列记录后继续后面的独立实验。

YOLO与classifier均只保留best/last（YOLO内置最终strip optimizer后会重写receipt）；失败run不可复用为另一个实验。固定8.4.189版本的首epoch OOM自动减半被显式关闭：显存不足使本实验失败，队列再继续其他候选；不会把requested batch8悄悄记成实际batch4。成功run另保存actual batch/nominal accumulate/最终trainer args。每个checkpoint的receipt绑定字节哈希、类别与数据/代码/配置签名。依赖失败时，cascade eval跳过；队列继续后续无依赖候选。不会把一个阶段失败误报为完整流水线成功。

## 准备与运行命令

在工程根目录，以Windows独立`mahjong-yolo`环境执行。以下命令是手工运行单项的例子；串行suite另由`infra/windows/prepare_experiment_suite.py`生成，本页命令没有被执行训练。

```powershell
python -m ml.data.prepare_yolo --dataset D:/mahjong/data/prepared-v4-main --out D:/mahjong/data/prepared-v4-main-yolo42
python -m ml.data.prepare_yolo --dataset D:/mahjong/data/prepared-v4-main --out D:/mahjong/data/prepared-v4-main-yolo1 --generic
python -m ml.data.prepare_crops --dataset D:/mahjong/data/prepared-v4-main --out D:/mahjong/data/prepared-v4-main-crops42
```

每个准备/训练命令均支持`--preflight-only`，不建run、不训练；严格检查现有衍生数据，`derivativeReady:false`表示还需准备。

```powershell
python -m ml.train.yolo_train --dataset D:/mahjong/data/prepared-v4-main --prepared D:/mahjong/data/prepared-v4-main-yolo42 --weights D:/mahjong/models/yolo26m.pt --output D:/mahjong/runs/yolo26m960-new --resolution 960 --epochs 50 --batch 4 --preflight-only
python -m ml.train.cascade_train --dataset D:/mahjong/data/prepared-v4-main --crops D:/mahjong/data/prepared-v4-main-crops42 --weights D:/mahjong/models/efficientnet_v2_s-dd5fe13b.pth --output D:/mahjong/runs/effv2s320-new --resolution 320 --epochs 35 --batch 16 --gradient-accumulation 2 --preflight-only
```

移除`--preflight-only`才实际训练。classifier仅接受224/320/384；YOLO输入为32的倍数。训练强制CUDA/BF16，不隐式退回CPU。离线模式关闭自动安装、云端日志和字体/AMP探测权重下载；GPU不支持BF16时明确失败。

```powershell
python -m ml.eval.yolo_evaluate --dataset D:/mahjong/data/prepared-v4-main --run DETECTOR_RUN/run.json --checkpoint DETECTOR_RUN/train/weights/best.pt --out DETECTOR_RUN/validation
python -m ml.eval.cascade_evaluate --dataset D:/mahjong/data/prepared-v4-main --detector GENERIC_RUN/train/weights/best.pt --detector-run GENERIC_RUN/run.json --classifier CLASSIFIER_RUN/best.pt --classifier-run CLASSIFIER_RUN/run.json --out CASCADE_EVAL_RUN/validation
```

正式suite使用候选比较器，避免只选择AP最高检测checkpoint或oracle crop分类最高checkpoint。它对best/last先按SHA去重，直接检测最多2个候选，cascade最多4对，然后按整手指标选。一个候选评估异常会记录失败并继续其他候选；全部失败才将eval job记失败。

```powershell
python -m ml.eval.evaluate_yolo_candidates --dataset D:/mahjong/data/prepared-v4-main --run DETECTOR_RUN/run.json --checkpoint DETECTOR_RUN/train/weights/best.pt --out DETECTOR_RUN/validation
python -m ml.eval.evaluate_yolo_candidates --dataset D:/mahjong/data/prepared-v4-main --run GENERIC_RUN/run.json --checkpoint GENERIC_RUN/train/weights/best.pt --classifier-run CLASSIFIER_RUN/run.json --classifier CLASSIFIER_RUN/best.pt --out CASCADE_EVAL_RUN/validation
```

评估只能读取完整已结束run；每次新建输出目录。`report.json`的`selectedCheckpoint`/`selectedClassifier`是最终候选路径。打包时应使用这两个路径，不能默认best一定胜出。`freeze_yolo`将验证报告所指向的精确checkpoint打包，保留`productionValidated:false`，不自动替换线上版本。下面单checkpoint示例仅适用于报告确实选择best时。

```powershell
python -m ml.train.freeze_yolo --run DETECTOR_RUN/run.json --checkpoint DETECTOR_RUN/train/weights/best.pt --report DETECTOR_RUN/validation/report.json --version yolo26m960-v4-candidate --out D:/mahjong/packages/yolo26m960-v4-candidate
python -m ml.train.freeze_yolo --run GENERIC_RUN/run.json --checkpoint GENERIC_RUN/train/weights/best.pt --classifier-run CLASSIFIER_RUN/run.json --classifier CLASSIFIER_RUN/best.pt --report CASCADE_EVAL_RUN/validation/report.json --version cascade-v4-candidate --out D:/mahjong/packages/cascade-v4-candidate
```

初次导出保留PyTorch native，先保证与训练结果一致。正式候选通过后再比较FP16 ONNX/TensorRT的预测逐图一致性、整手准确率、冷启动和成本；不预先使用INT8压低纹理精度。YOLO/cascade adapter及manifest校验已经准备，当前网站继续使用RF候选，未切换YOLO。

## 官方资料与许可

- [YOLO26官方模型与原生端到端模式](https://docs.ultralytics.com/models/yolo26/)
- [YOLOv8官方模型](https://docs.ultralytics.com/models/yolov8/)
- [训练API、Windows单GPU入口](https://docs.ultralytics.com/modes/train/)
- [8.4.189包元数据](https://pypi.org/project/ultralytics/8.4.189/)
- [官方v8.4.0预训练资产](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
- [Torchvision0.23 EfficientNetV2-S](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html)

Ultralytics包与预训练模型为AGPL-3.0，当前仅加入实验准备。公网产品部署前需按官方许可条款处理开源或商业许可，不能把私有仓库视为天然满足许可。Torchvision代码BSD-3-Clause，预训练权重另需遵循其训练数据相关条款。没有购买任何许可或调用付费训练。

四份官方预训练已完整下载，来源、字节长度和SHA256冻结在[yolo-cascade-pretrained.json](./yolo-cascade-pretrained.json)。`infra/windows/cache_yolo_weights.py`可重新缓存并校验Torchvision公开哈希前缀。模型加载与训练完全使用本地路径，不依赖训练中再下载预训练模型。

classifier的`--rotation quarter-turn`（默认）额外在训练时随机转0/90/180/270度，旋转完整正方形裁图，不镜像、不裁丢边缘；`--rotation none`可做消融。该设置写入run签名，valid与线上不随机旋转。
