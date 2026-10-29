# V7：剩余数据改进对照、小分类器、蒸馏与量化

R2执行版本：Windows现已开机，用户已另行授权优先提交最多12小时训练。实际同步、CPU预检、提交状态、包SHA和固定截止时间以本机动态ready/launch签名收据（不随公开源码分发）为准。本说明是冻结执行指南，不代替实际启动收据。现有V5/V6队列及第一版V7 ZIP保留；R2使用全新代码、suite及队列目录，不覆盖旧历史。

## 已知依据及目标

V6首组12轮25分钟完成；与同CPU、同Eff-last的V5父级相比，82张10+牌照片自动整手全对70→71，模型全对72→74，13/14牌自动43/45→42/45。背景误检、小牌漏检及检测置信导致的拒识仍存在，不能凭训练loss下降认定泛化改善。剩余7组还没有效果证据，保留其受控对照，不能宣称它们已经改进准确率。

本轮小模型目标是降低**第二阶段分类**开销，不改变已选V26定位器。MobileNetV3-Large224分别用监督CE和CE+KD训练，初始权重、样本、增强、seed、batch和20轮上限一致，只有KD项及预算不同。教师是当前已上线V26+EffV2S cascade中的EffV2S分类阶段，固定原始Eff-last签名权重；不重新训练教师、不宣称完成定位器蒸馏。真实整手验证使用检测器预测框，GT crop top1仅作训练诊断。

KD温度2、alpha0.5。仅当教师在该训练crop上的预测类等于GT且未温度化概率≥0.7时混入KL；其他crop完整保留GT CE，避免用教师错误替换人工标签。教师eval、requires_grad=False、no_grad，学生与教师重复同一原图坐标crop抖动、光照随机因子和整牌90°旋转，分别224/320字母框缩放，不中心裁掉牌字。教师只处理train，valid使用真实标签，不生成训练伪标签。

## 队列顺序与预算

| 顺序 | 组别 | 估算分钟 | 训练软预算／组硬超时分钟 | 目的 |
| ---: | --- | ---: | --- | --- |
| 1 | v8-background | 42 | 45／68 | 与已完成V8控制组相比增加60张审核负样本 |
| 2 | v8-background-geometry | 45 | 45／70 | 同一父级，单独加入轻几何 |
| 3 | student-ce | 40 | 40／65 | MobileNetV3-Large224监督对照 |
| 4 | student-kd | 55 | 55／80 | 同一学生，冻结EffV2S教师蒸馏 |
| 5 | student-ce-int8 | 8 | 无训练／18 | 对监督赢家做CPU FX INT8校准及冻结阈值验证 |
| 6 | student-kd-int8 | 8 | 无训练／18 | 对KD赢家做相同量化对照 |
| 7 | v26-background-geometry-960 | 50 | 60／85 | 当前较优检测架构的数据改进 |
| 8 | classifier-control | 75 | 80／110 | EffV2S warm-start基线 |
| 9 | classifier-mild-camera | 80 | 85／115 | 控制后的轻倾角／反光对照 |
| 10 | v26-background-geometry-1280 | 80 | 85／120 | 测量更高分辨率对密集小牌的收益 |
| 11 | classifier-context002 | 75 | 80／110 | crop上下文0.02与0.08对照 |

总估算558分钟，另保留60分钟，618<720分钟；这是估算，不保证11组全部跑完。7个V6剩余分支使用新ID／新输出，不重复已完成V8控制组，不覆盖旧pending收据。

轮数上限：7个V6分支各12轮，两个学生各20轮；INT8两组不训练。唯一同主机实测参考是V8控制组`1504.844秒÷12=125.404秒/轮`，即约2.09分钟/轮、12轮约25.08分钟。对同架构增加60张负样本后的42／45分钟估算留有余量；V26、Eff、MobileNet和KD没有据此实测，不能把它们都套成2.09分钟/轮。学生20轮是上限，不是保证跑满20轮；40／55分钟软预算会在已完整保存的轮次处停止。CPU准备／校验先于提交完成，队列12小时从协调进程创建窗口起算，包含训练、排队等待、整手验证和量化步骤。

训练进程在验证和原子权重保存完成后检查软预算，保存best/last、receipt、metrics.csv及实际完成轮数；预算完成不等于收敛。组硬超时、步骤超时和队列12小时壁钟+单调时钟上限同时生效，截止后只清理自己创建的进程树，未开始项目标`not_started_budget`。普通异常／OOM／超时保留状态并转下一独立组；两个INT8组仅依赖自己的学生训练+完整验证成功，失败则跳过对应量化，不拖住其他独立训练。

已验证桌面程序的完整路径沿用上轮参考，但Windows更新可能改变路径。未知GPU进程仍会阻止启动并在15分钟后blocked，而非擅自杀用户进程或自动放宽白名单。这类阻塞及关机／断电无法由训练队列保证自动恢复；新窗口必须重新确认机器状态和签名路径。重新启动同一队列不会延长原12小时截止。

## INT8范围与防泄漏

使用已锁定Torch2.8／Torchvision0.23内置FX static PTQ、fbgemm x86 CPU。不是GPU TensorRT INT8，也不直接部署到Modal。对每个学生，先用best/last全150张valid选出FP32赢家，再仅从train缓存确定性分层选672个crop（覆盖42类，优先各类不同源组）校准。validation和sealed test都不能用于校准。

量化后按FP32赢家的同一检测阈值及0.7接受阈值，对150张完整valid重跑，不重新找更有利的阈值；记录整手正确数差值、拒识、完整逐图预测、量化卷积数、artifact大小、校准记录及所有源权重SHA。FX不能支持当前模型／fbgemm不可用／未真正量化卷积时该组失败，不能静默切成另一个方法，也不能仅凭文件名称为INT8。

Windows上的新学生显存、训练速度和FX导出尚未实测。CPU INT8不一定更快，准确率可能下降；定位器、网络和Modal冷启动也不会因此消失。不能把Mac CPU算子可用性当成Windows GPU或x86 fbgemm验收。

## 离线资源及可复现入口

新增官方预训练文件：`mobilenet_v3_large-5c1a4163.pth`，22,132,113字节，完整SHA256：`5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`。来源、版本、许可证及本机文件路径见`docs/models/student-assets-v7.json`。运行时禁止自动下载；必须先把该文件复制到`D:/mahjong/models/`并通过完整SHA校验。

本地生成准备包（只生成，不运行）：

```bash
python -m infra.windows.prepare_efficiency_suite_v7 \
  --out artifacts/ready-suites/v7-12h-r2 \
  --bundle-out artifacts/ready-suites/code-v7-ready-20261029-r2.zip \
  --code-root D:/mahjong/code-v7-ready-20261029-r2 \
  --queue-id v7-efficiency-12h-20261029-r2
```

代码包不含模型权重和数据；附有SHA清单及独立verifier。Windows运行端先校验解包到全新R2目录、复制新增官方权重，再依次运行以下CPU准备／检查命令；实际完成情况见实时收据：

```bat
D:/mahjong/code-v7-ready-20261029-r2/prepared-suites/v7-12h-r2/prepare.cmd
D:/mahjong/code-v7-ready-20261029-r2/prepared-suites/v7-12h-r2/check.cmd
```

它们只验证源文件、train/valid像素／标签、缓存schema2、父权重签名、完整包版本和20个CLI参数边界，不启动训练。现有独立mahjong-v6环境可复用，不更改已冻结包版本。CPU准入通过且当前GPU进程满足既有保护规则后，使用已获用户单独授权的提交命令。是否已经执行必须看实际收据，不能仅因命令已生成而称已启动：

```bat
D:/mahjong/code-v7-ready-20261029-r2/prepared-suites/v7-12h-r2/submit.cmd
```

监看命令：

```bat
D:/mahjong/code-v7-ready-20261029-r2/prepared-suites/v7-12h-r2/watch.cmd
```

其中只有submit.cmd含`--start`。生成脚本及测试不会调用Windows提交命令；现场执行状态单独写入实时收据。正式训练的核心接口是`ml.train.student_train_v7 --preflight-only`（仅数据／签名预检），`ml.eval.student_evaluate_v7`（实际预测框best/last整手比较），`ml.train.student_quantize_v7`（train校准、导出、冻结工作点比较）。运行`--help`可看完整参数，准备suite已冻结每条精确argv。

## 验收条件

- 比较同一validation annotation与image-tree SHA，主指标82张10+牌自动整手全对，另报model exact、接受率、接受后准确率、13/14牌及全部150张。
- oracle crop top1不能替代真实检测框整手指标；失败和拒绝保持在分母。报告绑定检测器、分类器、crop几何、训练数据和代码签名。
- 量化必须在同一冻结工作点对照；CPU时间不能直接比较GPU线上4秒耗时，冷启动另计。
- 本轮不打开sealed test。未来候选即使valid领先，也需要相同设备的GPU/CPU推理合同验收；不会从本轮队列自动发布。

## 本地验证记录

2026-10-29准备时，基础工具环境运行114项ML测试（其中4项需要Torch的测试在此环境跳过）及42项Windows工具测试通过；另在独立Mac CPU环境Torch2.8.0／Torchvision0.23.0执行这4项Tensor测试全部通过，包括官方权重严格加载→42类CPU前向、蒸馏门控算术、真实框adapter融合、FX INT8转换／脚本化及变长batch CPU前向。测试没有优化器step、反向传播、训练模型或教师前向。

新套件的20个CLI解析边界和使用合成带签名数据的实际student预检通过；测试证明预检不导入Torch、不创建训练输出，也不打开哨兵sealed test。12小时换算、全局截止终止自有子进程、壁钟回拨不能延时、失败继续和依赖跳过均有CPU子进程测试。本地INT8接口测试使用Mac支持的qnnpack，不能据此宣称Windows fbgemm或GPU已验收。此段记录初次离线验证；R2真实Windows环境、完整缓存CPU预检及后续首轮训练的实际结果见实时收据，区分“准备通过”“已提交”和“首轮验证”。

## 官方依据

- [Torchvision0.23 MobileNetV3源代码与V2权重地址](https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/mobilenetv3.html)：选择本地V2权重构建学生，替换42类输出头；保留本项目整牌预处理。
- [PyTorch知识蒸馏教程](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)：soft-target KL的温度平方缩放及CE对照；本项目额外使用GT一致性和教师概率门控。
- [Torch2.8 prepare_fx](https://docs.pytorch.org/docs/2.8/generated/torch.ao.quantization.quantize_fx.prepare_fx.html)：PTQ先插入观察器，再以train crop校准、转换；这是CPU研究路线。
- [Torchvision0.23量化MobileNetV3源代码](https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/quantization/mobilenetv3.html)：量化与浮点模型的实现差别意味着必须做实际可用性和准确率比较，不能套用官方ImageNet量化准确率当麻将结果。
