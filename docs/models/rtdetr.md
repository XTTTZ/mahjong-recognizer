# RT-DETRv2 R18 对照路线

2026-10-26：已实现42牌面/旧43类数据契约、训练/推理适配、输入几何对照和训练日志。Mac通过16项相关CPU测试，并以faster-coco-eval1.8.0校验完美预测AP=1、空预测AP=0。Windows GPU smoke和实际识别成绩以主任务运行记录为准；这些CPU检查不证明GPU兼容或识别准确率。

## 固定来源与环境

- 官方权重：[PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd)，模型卡Apache-2.0，固定revision `5650961749fa93567c0d46fc7f43ea4f9e914107`。
- [Transformers RT-DETRv2文档](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)；此次按实际Windows锁定版复核[5.20.2处理器源码](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/rt_detr/image_processing_rt_detr.py)与[模型源码](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py)，使用`RTDetrImageProcessor`、`RTDetrV2ForObjectDetection`及同一个后处理器。
- 算法来源：[作者RT-DETR项目](https://github.com/lyuwenyu/RT-DETR)。本工程使用HF实现，不使用Ultralytics。
- 已验证Windows环境`mahjong-train`：Python3.11.17、torch2.8.0+cu128、torchvision0.23.0+cu128、transformers5.20.2、scipy1.16.2、safetensors0.8.0、Pillow11.3.0、numpy2.2.6、faster-coco-eval1.8.0。无需FastAPI、额外CUDA扩展或替换CUDA包；原始源config设`disable_custom_kernels=true`。

Mac的`artifacts/models/pretrained/rtdetr-v2-r18vd/`已保存config、processor、80,904,640字节safetensors及`source.json`。权重SHA256为`d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115`，已对照官方LFS摘要；两个JSON也已核验。用`--source-directory D:\mahjong\pretrained\rtdetr-v2-r18vd`离线加载并重新校验三文件，不会偷偷访问Hub最新版。未提供该参数时，仅下载固定revision的三个文件并验证同样摘要。

## 数据与输入

训练只打开`train/`与`valid/`，不打开test。类别必须为明确的42种牌面（原ID1–42，不含back），或历史43类（ID1–43，含back）。模型0起始类别、head大小和中文名均按实际schema派生；42类路线不会留一个从未训练的back head。父类别不得混入。

训练前核验来源分组、完整train/valid实际图片字节、尺寸、EXIF、框界限、COCO和数据清单hash。运行签名绑定训练/适配器以及公共metrics、operating_point、postprocess、manifest、provenance代码摘要，另绑定数据、类别、训练参数、预训练资产和实际torch/transformers/numpy/Pillow/scipy/faster-coco-eval/safetensors版本。运行时版本仅在GPU路径读取，无GPU预检仍不导入torch。图片真值必须先规范化到EXIF=1；数据加载器的八方向图/框变换函数另有像素对齐测试，不会只旋转图片而不旋转框。

`--input-mode warp`保留官方固定方形resize基线。`--input-mode letterbox`先按比例缩放居中填充灰边，再进入同一个已保存处理器；图和COCO框一起变换。推理与验证通过同一个`prepare_input`和`restore_tiles`映射回原始图片坐标，记录`model.json.inputMode`。圆整后的x/y缩放分别保存，避免长条手牌在非整数尺寸时累积坐标误差。不能只改线上预处理或输入尺寸，却沿用另一种训练结果宣称等价。

没有镜像、随机裁剪或隐含几何增强。`--augmentation photometric`仅对train使用温和的亮度0.8–1.2、对比度0.85–1.15、色彩0.9–1.1；12%概率轻模糊半径0.2–0.7、20%概率JPEG质量75–95。框和画布不改变，valid不增强。变换在DataLoader worker执行，种子/worker RNG纳入已有恢复状态；不增加第三方依赖。`none`为无增强对照。

## GPU smoke与队列配置

先完成无GPU预检（不会访问网络、写输出或读取test）：

```text
python -m ml.train.rtdetr_train --dataset <prepared-v4> --output <new-run> --source-directory <verified-r18-source> --data-manifest <v4-manifest> --preflight-only --verify-images
```

再用32张train/8张valid、1epoch测试真实模型的前后向、标签loss、AMP、保存和加载：

```text
python -m ml.train.rtdetr_train --dataset <prepared-v4> --output <new-smoke-run> --source-directory <verified-r18-source> --epochs 1 --train-limit 32 --valid-limit 8 --batch-size 1 --accumulation 2 --workers 0
```

完整对照使用相同数据与有效batch16，先串行测显存和每epoch时间，再安排5–24小时队列。建议优先顺序如下；小时数由实际GPU smoke和全数据步数估算，不按空转凑时长。

| 实验 | 参数差异 | 要回答的问题 |
| --- | --- | --- |
| R18 / 640 / warp / none | physical batch4，累积4 | 和RF相比的独立架构基线 |
| R18 / 960 / warp / none | physical batch2，累积8 | 同输入策略下高分辨率是否改善小字和多牌 |
| R18 / 640 / warp / photometric | 与首组仅增强不同 | 温和外观增强对光照/模糊泛化的收益 |
| R18 / 960 / letterbox / photometric | 预算允许时增加；几何和增强均有变化 | 长条输入保形路线的整体候选收益，不能把变化只归因于一个因素 |

默认AdamW：普通参数1e-4、backbone1e-5、weight decay1e-4、梯度裁剪0.1、5%warmup加cosine。fp16需要用实际loss和梯度确认稳定，不能看到库可导入就假定兼容。workers0先跑通，之后workers2改善图片解码/增强吞吐。640和960的batch需以4070Ti12GB实测峰值为准，禁止两个训练同时抢GPU。

`--min-epochs 10`防止训练尚未收敛就被早期平局结束；可按完整run预算调高。`--validation-interval 2`可隔轮验证，但首轮和最后一轮始终验证。`--patience`计数的是无改善的**验证轮次**，不是所有train epoch；只有超过min-epochs且patience耗尽才早停。数据不足或val太小不能靠调大patience证明准确率。

## 指标、产物与恢复

- `metrics.csv`逐epoch记录loss、学习率、步数、耗时、整手exact/接受覆盖率以及mAP。没有验证的epoch对应列留空，不伪造或重复上一轮数字，通用队列监控器可以读取。
- 验证使用相同API输出规则（严格`score > threshold`、裁边、圆整、uncertain与拥挤拒绝），主指标按完整数量/类别/定位匹配，拒识不从分母移除。检测阈值0.15/0.25/0.35/0.5/0.65仅在valid选择，自动接受阈值默认0.7。
- 报告分别保存所有图片、≥10牌、`raw_image_research`与`oracle_roi_auxiliary`子集。按真值框裁出的ROI始终显式区分；raw子集也不自动升级为独立现场测试。
- [faster-coco-eval官方API](https://mixaill76.github.io/faster_coco_eval/examples/eval_example.html)计算标准bbox AP@100（mAP50、50:95、75），它是补充诊断。候选优先整手exact，再用原始model exact、接受覆盖率和mAP打破早期平局，不用单张分类准确率代替整手。
- `validation-NNN.json`保存每次验证；`best/`包含safetensors、config、processor、标签、hash清单与预测报告。候选始终`productionValidated=false`，`partialValidation`标明是否仅跑了smoke子集。
- 仅保留原子替换的完整`last.pt`和当前`best/`，不每epoch另存几百MB权重，避免多组长训练占用几十/上百GB。所有逐轮轻量指标仍保留。
- `last.pt`含模型/optimizer/scaler/scheduler、Python/numpy/torch/CUDA和DataLoader RNG、processor/config、epoch、最佳分数与运行签名。恢复只重跑未完成epoch；相同CLI加`--resume <trusted-last.pt>`，修改数据、代码、epochs、增强或输入模式应新建实验。恢复到新目录时要求上个last.pt旁保留best/，会复制历史最佳；完整checkpoint只接受本工程可信文件。
- CUDA算子可能非确定；保存RNG不代表跨GPU、跨版本逐bit复现。若已有早停检查点，不会无提示地重新开启同一实验。

## 推理

`RTDETRPredictor(best_dir)`在本地核验config/processor/safetensors、类别及阈值后读取，禁止隐式Hub访问。保存的warp/letterbox策略与训练一致，输出输入PIL图片像素中的XYWH。HTTP层已EXIF转置，适配器不再重复转置。

Modal通用工厂已支持此adapter，云镜像和正式发布仍需实际候选完成、完整验证、独立手机拍照/选区评估、导出一致性与冷/热费用测试。本文件的CPU准备状态不能视为上述验收已完成。
