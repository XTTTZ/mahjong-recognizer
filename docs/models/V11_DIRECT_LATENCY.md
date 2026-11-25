# 单阶段手机识别实验 V11

YOLOv8n 在同一前向计算中定位并识别 42 类牌面。三组输入尺寸为 640、800、960，先运行 640。与 V9 的单类别检测器相比，这三组不调用逐牌 MobileNet 分类器。输入尺寸改变计算量，三个模型的参数量基本相同。最终浏览器下载字节数须实测导出的模型。

| 输入边长 | 训练软上限 | 整手验证 | 配对 CPU 测速 | 单组硬上限 |
| --- | ---: | ---: | ---: | ---: |
| 640 | 135 分钟 | 30 分钟 | 25 分钟 | 195 分钟 |
| 800 | 170 分钟 | 30 分钟 | 25 分钟 | 230 分钟 |
| 960 | 190 分钟 | 30 分钟 | 25 分钟 | 250 分钟 |

各组独立从固定 SHA256 的官方 COCO 权重开始。最多 60 epochs，batch 8、seed 2026，AdamW、余弦学习率、3 轮预热、初始学习率 0.0003，轻微颜色及几何增强。训练软上限按完整 epoch 保存，硬上限留 5 分钟退出。三组合计最多 675 分钟，另留 20 分钟调度余量。协调器同时执行 12 小时总上限及新授权提供的绝对截止时间，任一较早即停止；单组失败后继续下一组。

## 数据和比较

使用 V4 主数据集中 2,629 张训练图、17,868 个牌框，冻结验证集含 150 张图和 1,396 个框。签名覆盖标注、像素、牌名顺序和模型初始权重。按相同的置信阈值 0.35、接受阈值 0.7，在全验证集比较已保存的 best/last，按已有整手排序选择。封存测试集保持关闭。

每组另产出 `paired-cpu.json`。在 Windows 同一 CPU 上比较当前 V10 已选权重的 PyTorch FP32 双阶段实现和该组单阶段模型。每图交替执行顺序，固定单线程，先预热，报告全部 150 张、至少 10 张牌的 82 张、13–14 张牌的 45 张的整手准确数、误接受数、p50/p95 和逐图差异。计时包含预处理、推理及输出归一化；磁盘、模型加载、图片解码及网络在计时外。该测速用于筛选，浏览器 WASM/WebGPU 和 Modal ONNX 要在各自部署环境复测。

候选应在 V10 容差内，相对同轮对照最多分别少对 3/150、2/82、1/45，各组最多多一次误接受，要求零失败。13–14 张牌热推理的目标为 p95 不超过 2 秒。优先对速度提高至少 20% 或导出文件缩小至少 30% 的候选做导出验证。在部署前检查实际浏览器模型大小、相同验证集的 ONNX 输出及真机速度。

## 11 月 25 日实测

三组各完成 60 轮，连同评估共 2 小时 17 分钟。下表为验证集自动整手正确的照片数。最后一列为正常 13–14 张牌的 Windows 单线程 PyTorch FP32 热推理 p95。

| 模型 | 全部 150 张 | 多牌 82 张 | 13–14 张牌的 45 张 | 正常手牌误接受 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V10 双阶段 | 137 | 74 | 44 | 0 | 201–208 ms |
| 单阶段 640，last | 72 | 40 | 30 | 4 | 69 ms |
| 单阶段 800，last | 84 | 45 | 32 | 2 | 96 ms |
| 单阶段 960，best | 88 | 47 | 35 | 2 | 124 ms |

保留 V10。960 在 82 张多牌照片里有 23 张输出牌数有误，V10 为 8 张。即使降低接受阈值，最多只有 6 张原本所有牌与框均正确的照片可从人工核对转为自动通过。模型选择与评估均在冻结验证集完成。

## 离线准备与以后启动

公开的[对照模型清单](./compact-v11-baseline.json)列出检测器、分类器及各自训练记录与权重收据的文件位置和 SHA-256。离线打包只需要这份清单。执行配对测速时，训练机上需备齐清单中的六个文件，程序会逐一校验。

在项目根目录创建一个新的不可覆盖计划和 ZIP：

```sh
python3 -m infra.windows.prepare_direct_suite_v11 \
  --out artifacts/runs/prepared-v11-direct \
  --bundle-out artifacts/ready-suites/v11-direct.zip
```

训练机需要现有的 mahjong-v6 环境、V4 主数据、签名的官方 nano 权重和两份 V9 基线权重及签名。只在收到新的训练窗口后使用 `submit.cmd`。若 GPU 活跃，现有协调器使用独占检查。需要准许当前桌面程序时，先核实完整可执行文件路径，然后用 `--gpu-background-paths` 和新的输出名重建包。新的队列 ID 使用 `--queue-id`，新代码根目录使用 `--code-root`。

把 ZIP 及同名 `.verify.py` 传到 Windows 的 `D:\mahjong\incoming`，从本机受信任的 `.receipt.json` 读取 `sha256`。在 Windows PowerShell 执行以下命令，先把环境变量设为准确的散列值：

```powershell
Set-Location D:\mahjong\incoming
& (Join-Path $env:USERPROFILE 'miniforge3\envs\mahjong-yolo\python.exe') .\v11-direct.verify.py unpack --bundle .\v11-direct.zip --expected-sha256 $env:V11_SHA256 --destination D:\mahjong\code-v11-direct-20261125
Set-Location D:\mahjong\code-v11-direct-20261125\prepared-suites\prepared-v11-direct
.\prepare.cmd
.\check.cmd
```

提交要求提供新授权的、包含时区的 ISO 8601 截止时间。将其填入环境变量，签名检查与预检耗时也占用授权时间。空值、无时区、已过期或超出 12 小时的值都会被拒绝。相同队列目录不会再次获配新的窗口。

```powershell
.\submit.cmd $env:V11_AUTHORIZED_UNTIL
.\watch.cmd
```

`watch.cmd` 显示当前组、步数、epoch、loss、mAP、剩余时间。队列收据位于 `D:\mahjong\queues\v11-direct-latency-12h-20261125`，每组的 `metrics.csv`、`validation/report.json` 和 `paired-cpu.json` 在 `D:\mahjong\runs` 的对应子目录。运行记录中的 Windows 预检、训练和实测速率与该次提交绑定。
