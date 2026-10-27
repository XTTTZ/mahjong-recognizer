# 多路线实验：准备完成后由用户手动提交

本轮只准备代码、环境、数据衍生物、预训练资产与启动命令，**不提交模型训练**。正式代码目录约定为`D:/mahjong/code-v5-ready-20261026`；最终校验结果和冻结时间由交付时的suite及CPU preflight receipt确定。单有本文不等于Windows上的所有准备已经通过。

## 比较什么

优先指标为同一张完整输入照片的**全部牌均正确**，必须同时包含定位漏检、多检、类别混淆、拒识与请求失败。单牌crop分类准确率只用于诊断；两阶段的最终分数必须由真实检测框裁剪后分类计算。所有方案使用相同的共享阈值策略、后处理及整手计分，不强制补足14张牌。

- YOLOv8s直接42类检测复现熟悉模型的检测基线；YOLO26m直接检测检验较新架构的实用收益。
- YOLOv8s/YOLO26s各训练一个单类“正面牌”定位器，共用同一份EfficientNetV2-S分类器。这样可比较旧路线与新定位器，并将定位错误与图案分类错误分开诊断。分类器输入320px，完整牌面按比例缩放和padding，保留8%上下文；训练加入轻微框抖动，适应实际检测框偏差。
- RF-DETR Small的640无增强→640光照增强→960光照增强，逐项检查增强和分辨率收益。
- RT-DETRv2 R18的640warp→960warp→960letterbox，逐项检查分辨率和几何预处理收益。
- RF Medium/960、Large/768作为速度、内存与整手准确率的联合候选；这些不是只改变架构的因果对照。
- RF Small/960扩展数据作为预增强覆盖数量消融：train2629→4671，验证集保持字节一致。扩展版不是增加2042张独立实拍。

用户原先YOLOv8s+EfficientNet的历史实现与准确率未作为可比的冻结实验导入；这里是在统一数据、输入及整手计分下重新建立基线。模型版本、官方预训练来源与许可见[YOLO与两阶段](YOLO_CASCADE.md)，不因模型较新就预先判定它更准确。

## 数据与评估边界

主数据train2629/valid150/test121，训练框17868，valid有82张≥10牌照片；扩展训练版4671图/31545框。42类正面牌均有实例，背面视为背景。分组隔离同原图增强和推断出的近邻拍摄序列；原始test来源家庭不回流train。训练、预检及架构选择只打开train/valid，sealed test保留。

这些图片仍来自拼合的公开数据，含原图的既有增强；主训练集约372份原图、263个保守分组。validation的来源和场景也有限，因此研究验证分数不能直接称为棋牌室实拍准确率。未标注的257张Kaggle手机照片仍独立暂存，未混入本轮训练或验证。

## 预排目录

按顺序串行执行；预估共**33.5小时**，有意准备超过单次硬预算的候选。5/12/24小时使用各自独立队列ID、运行目录和截止时间，**不会承诺一次全部跑完**。每组开始前检查其估时加15分钟预留是否能放入剩余预算；放不下则记录跳过，继续检查后面的较短候选。估时只是旧小批probe的粗略外推，首次全量实测后需要更新。

| 顺序 | 路线 | 输入 | 最多epoch | 预估/组硬超时（分钟） |
|---|---|---|---:|---:|
|1|YOLOv8s直接42类|960 letterbox|50|95 /140|
|2|YOLOv8s单类定位|960 letterbox|50|80 /130|
|3|EfficientNetV2-S分类|320整牌padding|35|55 /90|
|4|v8定位+同一分类器，端到端验证|真实检测框|—|10 /25|
|5|YOLO26m直接42类|960 letterbox|50|130 /190|
|6|YOLO26s单类定位|960 letterbox|50|80 /130|
|7|v26定位+同一分类器，端到端验证|真实检测框|—|10 /25|
|8|RF Small，无增强对照|640 warp|35|110 /165|
|9|RF Small，光照增强|640 warp|35|115 /170|
|10|RF Small，光照增强|960 warp|35|170 /235|
|11|RT R18，光照增强|640 warp|35|110 /170|
|12|RT R18，光照增强|960 warp|35|165 /225|
|13|RT R18，光照增强|960 letterbox|35|165 /225|
|14|RF Large，光照增强|768 warp|30|190 /255|
|15|RF Medium，光照增强|960 warp|30|230 /300|
|16|RF Small，扩展train，光照增强|960 warp|30|250 /315|
|17|YOLO26s直接42类，较短备用候选|640 letterbox|25|45 /90|

训练可以早停，因此实际耗时也可能显著缩短。组超时涵盖训练和评估，评估step另有限时。训练崩溃或超时后直接进入下一个独立模型，**不会把部分checkpoint或未完成评估当成成功**。两阶段评估依赖定位器与分类器两个job均成功；任一失败则记录`skipped_dependency`，继续其他独立实验。超时可能正在写checkpoint，恢复前要显式校验完整性。

每组保存独立`run.json`、`metrics.csv`、checkpoint、队列status与日志；验证报告按相同valid标注SHA和图片tree SHA汇总。RF选择最多2份不同checkpoint验证；YOLO直接检测对best/last按SHA去重后最多比较2份，两阶段对定位器best/last与分类器best/last最多比较4种配对，均在完整valid上重新计算真实检测框的整手分数，再按整手exact→model exact→coverage选优。每个候选保留签名、权重SHA和报告；子候选失败留档后继续，全部失败才令本组评估失败。RT按共享整手指标保存最佳。RF库内mAP和标准COCO AP@100的上限设置不同，主要架构选择看统一整手报告，不能直接混称全部mAP数字完全同定义。

## 冻结和准备（维护者）

先完成两个Windows环境：RF/RT用`mahjong-train`，YOLO/分类用隔离的`mahjong-yolo`。各自捕获新`environment.json`与requirements lock，不能覆盖旧probe的环境证据。官方资产清单使用以下结构：

```json
{"assets":[{"path":"D:/mahjong/models/yolov8s.pt","sha256":"完整SHA256"}]}
```

清单需覆盖3份RF、3份YOLO、1份EfficientNet权重以及RT的config、processor和safetensors。最终文件名由资产清单冻结。`prepare_experiment_suite`不会训练，不联网，生成配置也不会创建Windows训练目录。

代码、资产、数据和环境快照全部稳定后，为5、12、24小时分别调用（以下为24小时示例；其他两份更改`--hours`、`--queue-id`、`--out`，不能复用队列ID）：

```bash
python3 -m infra.windows.prepare_experiment_suite \
  --out artifacts/ready-suites/v5-24h \
  --hours 24 --queue-id v5-breadth-24h-20261026 \
  --code-root D:/mahjong/code-v5-ready-20261026 \
  --gpu-background-paths artifacts/runs/queue-prep/gpu-background-paths-current.json \
  --asset-manifest infra/windows/verified/assets-v5.json \
  --environment-snapshot infra/windows/verified/v5/train/environment.json \
  --environment-snapshot infra/windows/verified/v5/yolo/environment.json
```

把冻结代码包和生成的目录放到Windows，`v5-24h`需位于代码根的`prepared-suites/v5-24h`。`suite.json`的`preparationCommands`列出3条CPU数据衍生命令，分别生成42类YOLO、单类YOLO和lazy crop索引；不会打开test。先执行这些CPU准备命令，再运行`check.cmd`，或对应的Python命令：

```text
C:/Users/TRAINING_USER/miniforge3/envs/mahjong-train/python.exe D:/mahjong/code-v5-ready-20261026/infra/windows/launch_prepared_queue.py --suite D:/mahjong/code-v5-ready-20261026/prepared-suites/v5-24h/suite.json
```

不带`--start`只校验，检查包括源代码/metadata SHA、全部权重SHA、两个环境完整包集和`pip check`、train/valid真实图片、3份衍生数据、26个训练/评估step的参数解析，以及每条训练命令的CPU预检。预检停止在模型训练前，不分配训练任务。数据准备缺失、包版本漂移、旧run目录冲突或代码变化均拒绝启动。代码变更后需要重新冻结、生成新suite，不能修改旧队列历史冒充原计划。

## 用户手动提交与监看

本轮**不执行**下一条命令。准备通过后，用户在Mac终端提交24小时队列：

```bash
ssh win "C:/Users/TRAINING_USER/miniforge3/envs/mahjong-train/python.exe D:/mahjong/code-v5-ready-20261026/infra/windows/launch_prepared_queue.py --suite D:/mahjong/code-v5-ready-20261026/prepared-suites/v5-24h/suite.json --start"
```

选择5/12小时时替换suite路径为`v5-5h`/`v5-12h`，只启动其中一个。提交器先重做CPU校验，再通过WMI启动独立后台协调器；退出SSH、关闭聊天均不影响训练。无需Codex定时唤醒，也无需让本次对话sleep数小时。Windows必须保持开机；进程在运行期间临时阻止系统休眠，屏幕仍可关闭。断电、Windows重启或手动杀进程无法由内存中的协调器自动恢复，不能宣称绝对不间断。

持续查看loss、mAP、当前模型、GPU与剩余估时：

```bash
ssh win "C:/Users/TRAINING_USER/miniforge3/envs/mahjong-train/python.exe D:/mahjong/code-v5-ready-20261026/infra/windows/watch_queue.py --queue D:/mahjong/queues/v5-breadth-24h-20261026 --follow"
```

Ctrl+C只退出监看。去掉`--follow`只读一次，添加`--json`读取机器状态。不要把`state=completed`理解为全部训练成功，还应检查`result`、成功/失败/预算跳过/依赖跳过计数以及`summary.json`中的每组报告。

相同配置再次手动提交时，已有成功、失败、超时和跳过receipt不会重复训练；旧活进程只会等待，绝不按历史PID擅自杀进程。协调器异常后未完成job记录为`interrupted`，其余未处理job可继续，首次启动的24小时截止时间不会延长。RF/RT可在确认完整checkpoint、原签名和原目录符合合同后单独安排显式恢复；YOLO/分类器当前采用新的独立run重试，不能承诺无损自动续训。

## 已完成的工程验证

[历史小批probe证据](probes-v4-20261026.json)：5组1epoch GPU probe均完成；它们是本轮准备之前已有的结果。RF Small640/Medium960/Large768分别36.12/48.63/42.52秒，峰值6.61/3.63/2.87GB；RT640warp/960letterbox分别11.15/12.22秒，峰值1.21/2.11GB。RF只用96train/24valid，RT只用32train/8valid，不能据此比较准确率或可靠地预测整套耗时。

队列工具CPU测试覆盖失败后继续、训练后评估、只终止自有超时进程、依赖跳过、系统锁、重启幂等、预算跳过、WDDM进程边界、代码/权重/环境漂移拒绝、衍生数据缺失拒绝与所有prepared argv只解析不执行。最终Windows实际CPU预检是否通过，以交付receipt为准；这不等于新YOLO或两阶段的GPU训练验证。
