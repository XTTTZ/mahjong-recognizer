# 下一轮训练与发布准备

更新：2026-10-26。Windows已重新开机，清洗数据已在Windows重建核验，960分辨率2epoch smoke成功后已启动独立full run。以下保留可复现的准备与训练顺序；实际进度见[运行记录](./training-run-v3.json)与Windows状态文件。本轮尚未部署云GPU。

## 先冻结数据，再启动新实验

原始v2完整保留。其首个完整epoch发现混合标注范围后已停止，记录见[实验事实](./experiment-rfdetr-small640.json)。旧checkpoint只作历史研究证据，不接着训练清洗后的数据。

v3采用明确记录的排除决定、原始照片家族去重和人工审查ROI；若ROI由真值框计算，它是oracle-derived输入，只能用于训练与辅助验证。其成绩不能称为“用户拍照→自动识别”的端到端准确率。没有独立棋牌室手机测试集之前，不报告生产准确率达标。

离线代码包和数据准备入口见[Windows说明](../../infra/windows/README.md)。数据重建复用原始COCO ZIP与固定选择清单，核验图片、类别、框、来源组和hash。数据、阈值、模型文件与报告均保留原版本，改变数据后建立新run。

## 两条对照路线

| 实验 | 目的 | 当前实际状态 |
| --- | --- | --- |
| RF-DETR Small，640 | 复用已跑通的Windows GPU路径，建立干净数据基线 | 旧v2 smoke/首epoch已运行；新的签名与清洗数据流程只做了CPU检查，需再跑GPU smoke |
| RF-DETR Small，960 | 检查小字/多牌收益与12GB显存、耗时 | 需先测batch1/2显存，不预先承诺收益；不与640同时抢GPU |
| RT-DETRv2 R18，640 | 独立架构对照，避免只凭一个模型家族作选择 | 固定官方revision的权重/配置已预取并核验；Windows前后向尚未运行 |

当前两条原生处理器都固定方形resize，会改变长宽比；网页的长边压缩与手动裁剪不等于模型内部letterbox。是否增加成对训练/推理的letterbox实验，要由实际长条手牌验证决定。不能只改推理尺寸后宣称等同另训的高分辨率模型。

基线禁用镜像、随机裁剪和隐含几何增强，防止麻将字形翻转。清洗后小数据容易过拟合；后续优先增加独立实拍、确认标注，再比较受控光照/模糊增强。任何增强仅用于train，原始照片数与增强样本数分别报告。

## 不用GPU的预检

在工程根目录运行，`<prepared-v3>`与`<split-v3.jsonl>`替换成实际已重建路径：

```text
python -m ml.train.rfdetr_train --dataset <prepared-v3> --output <new-rf-run> --data-manifest <split-v3.jsonl> --preflight-only --verify-images
python -m ml.train.rtdetr_train --dataset <prepared-v3> --output <new-rt-run> --data-manifest <split-v3.jsonl> --source-directory <rtdetr-v2-r18vd> --preflight-only --verify-images
```

预检不加载torch、访问网络、打开test或写训练输出。完整检查会解码train/valid、验证尺寸/EXIF/框界限并计算实际图片字节摘要。元数据预览可省去`--verify-images`，但它不证明图片可读或字节未变。正式训练及恢复始终检查图片。

先用32–64图、1epoch验证前后向、标签、保存、加载及输出坐标，再执行完整run。RT具体参数与固定源权重见[RT-DETRv2说明](./rtdetr.md)。RF physical batch8/accumulation2仅在旧640 smoke观察到约6.2GiB峰值，不能直接套到960或新架构。

## 恢复和评估约束

RF完整epoch checkpoint含optimizer和scheduler，代码通过`args.notes`将其绑定到数据、图片字节、类别、训练参数和代码签名。只允许同一run目录恢复；轻量best权重不能当作恢复断点。RF库每次fit重新设seed，因此目前只声明恢复epoch/optimizer/scheduler；连续与中断训练的随机轨迹一致性尚未验证。

RT保存模型/optimizer/scaler/scheduler、Python/numpy/torch/CUDA及DataLoader RNG，但仍不保证跨库、跨GPU逐bit一致。其运行签名也包含实际图片摘要；修改配置、数据或代码应新建实验。

验证仅使用valid，分别报告所有图片、单牌、≥10张牌以及13/14张牌。数量、类别和定位需要一一匹配，重复牌不能掩盖漏牌；拒识和请求失败都留在主指标分母。阈值缓存与HTTP共用clamp、round、uncertain和拥挤拒绝规则，阈值边界采用检测器的严格`score > threshold`。

RF评估核对checkpoint内签名、实际架构/分辨率、训练run和完整数据字节摘要，拒绝把其他run的模型与当前报告拼接。打包还要核对权重与报告hash、全量valid、阈值来源；产物仍标记`productionValidated=false`。默认优先打包轻量best权重，完整`.ckpt`含训练状态，会增加上传、冷启动与内存。

## Modal与正式域名

`python -m ml.inference.model <frozen-model-directory>`可在无GPU环境检查43类映射、阈值、模型/processor/config文件hash；它不证明模型能在GPU执行。

Modal构建前运行同一检查，按manifest选择RF或RT适配器，固定主要推理依赖；下一步Linux镜像构建后还需保存完整依赖锁、做导出/精度一致性检查并测冷/热响应和实际费用。当前默认staging App、缩零、单容器和proxy auth。

最终地址已配置为`mahjong.avalonxty.site`。正式上线仍需真实模型、手机拍照/裁剪/上传联调、独立现场测试和成本核验。当前Worker dry-run与模拟响应测试只证明工程合同与界面，不能代替这些步骤。
