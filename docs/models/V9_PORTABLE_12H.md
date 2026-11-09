# V9：准确率不退步的12小时轻量化实验

准备日期：2026-11-06。目标是普通13/14张手牌热模型完整流水线p95≤2秒；定位、裁图、归一化、分批分类、后处理全部计时。首次下载、模型加载、冷启动、上传另报。用户随后已授权提交12小时窗口；实际启动/预检结果以独立运行回执为准。

## 已有证据与本轮选择

已复核 `artifacts/runs/portable-student-cpu-20261030.json`：同一Modal x86容器、2物理核、ORT1.23.2、batch4、交替运行、150张完整valid。现有Eff320与V7 CE224在82张多牌照片都是72张自动正确，45张13/14牌均43张正确。13/14牌p50/p95由1.797/1.946秒降到0.544/0.613秒，ONNX总大小119.09→55.33MB。这是热流水线，不含网络/首次加载，也不是手机数据。

CE224仍有两处退步：ID56七张牌因置信度被拒，完整集自动正确135→134；ID16十四张牌多出一张预测，两模型都拒绝，但模型本身整手正确下降。因此保留原线上模型，训练优先修复学生准确率，再压缩定位阶段。学生路径中定位约405ms，分类约94ms，定位已是主要耗时。原FX INT8明显掉准确率，不重复该配方。

## 串行训练队列

所有训练独立运行，不依赖前一组成功。训练达到轮数或完整epoch预算即停止，实际完成轮数入回执；表中不是收敛保证。

| 顺序 | 配置 | 轮数上限 | 训练软上限 | CPU整手验证上限 |
|---|---|---:|---:|---:|
| 1 | MobileNetV3-Large224，CE，context .08 | 40 | 50分钟 | 20分钟 |
| 2 | YOLO26s640，已有960定位权重微调 | 30 | 55分钟 | 25分钟 |
| 3 | YOLOv8n800，官方COCO初始化，新单类牌框 | 45 | 80分钟 | 25分钟 |
| 4 | MobileNetV3-Large224，轻蒸馏α=.2、T=2 | 40 | 90分钟 | 20分钟 |
| 5 | YOLOv8n960，与第3组匹配配方 | 45 | 95分钟 | 25分钟 |
| 6 | YOLO26s800，与第2组匹配配方 | 30 | 70分钟 | 25分钟 |
| 7 | MobileNetV3-Large192，CE，与第1组匹配配方 | 40 | 45分钟 | 20分钟 |

训练软上限合计485分钟，验证160分钟；每组另留5分钟退出宽限，全部任务硬上限合计680分钟，再留30分钟队列余量，共710分钟。全局720分钟硬截止包含验证、等待显卡等；故可能提前完成或末组因预算不足不启动，不为凑时长重复训练。旧V7 CE20epoch约24分钟、KD20epoch约46分钟是估时参考；nano的时间尚未实测。

定位batch8、BF16、seed2026；背景使用已人工审查的train负样本，轻几何±8度、平移.04、缩放.1、透视.0004，不做水平翻转、拼图。V26低学习率.00005微调；nano .0003、3epoch warmup，从官方小架构初始化，不能把s模型改文件名当n。学生batch16×累积2、lr.0003、seed2026、ImageNet官方初始化；蒸馏只使用GT类别正确且置信度≥.7的教师预测，人工标签仍为真值。

沿用2629训练照片/17868个牌框、150验证照片/1396框；没有新增独立数据。冻结82张≥10牌主评估及45张13/14牌切片，6张偏小/斜图单列压力参考，76张日常切片是事后切片，不声称新独立测试。封存test不打开。

## 质量门槛与训练后实验

每组best/last按生产点检测>.05、严格融合>.35、接受.7比较，不通过改阈值“补回”准确率。定位候选先固定Eff320；分类候选固定原960定位器，筛选后再组合。完整150图逐张保护原自动正确和原模型正确的照片，禁止新增错误接受，任何失败/不完整验证不算通过。整手指标之外单报定位P/R、同一组几何匹配框上的分类准确率。此门槛是保守工程选择，不是小验证集上的统计泛化证明。

已有研究代码支持有签名的YOLOv8n/v8s/26s640/800/960＋Eff320或MobileNet192/224导出FP32 ONNX，独立于线上格式：

```bash
python -m infra.portable.candidate_export --dataset DATASET --detector-run DETECTOR_RUN_JSON --detector-checkpoint DETECTOR_PT --classifier-run CLASSIFIER_RUN_JSON --classifier-checkpoint CLASSIFIER_PT --out FRESH_EXPORT --preflight-only
# 去掉 --preflight-only 才导出；未正常完成的旧父模型必须另传对应 --classifier-terminal-receipt。
python -m infra.portable.candidate_benchmark --baseline-dir FROZEN_BASELINE --candidate-dir FRESH_EXPORT --valid-dir DATASET/valid --out FRESH_REPORT.json
# 默认只验证；--run-local 才做本机推理，--run-cloud 才启动一个最多1800秒的Modal CPU评估任务。
```

导出使用独立便携环境，不能为ONNX修改已冻结的Windows训练环境。训练窗口只包含训练和CPU整手验证，不偷偷启动云端费用或手机基准测试。后续矩阵见 `v9-runtime-matrix.json`：

1. FP32先在同容器同批照片比较CPU1/2物理核、batch1/4/8；默认先2核batch4，再把优胜者降到1核。分别记录常见手牌与大图的p50/p95、失败/内存、加载/冷启动，不能用Windows时间代替Modal。
2. 真机iPhone Safari、主流Android Chrome测WASM/WebGPU，明确首次下载、缓存后加载、首张及连续10张、发热/后台恢复。每轮最多3次由用户继续，照片留本地。桌面模拟手机宽度不是手机速度。新的候选格式必须先适配独立benchmark页面，现有页面仍仅接受原冻结profile，不自动上架新模型。
3. FP32达标后才做单阶段ONNX QDQ静态INT8：train分组校准，分类覆盖42类，使用真实定位裁图；先分类量化、再定位量化，比较中间输出，敏感层保持浮点。FP16优先WebGPU/TensorRT，不能假设CPU FP16更快。这些是训练后受质量门槛约束的实验，当前没有声称已完成QAT或量化验证。
4. 若CPU/真机达标，优先CPU云端或用户选择的本地推理；否则对比L4/T4，保持原A10回退。TensorRT引擎要在目标卡重建，不能复制A10引擎并假定兼容。每条路线都复核整手正确和冷启动。

## 费用边界

2026-11-06核对[Modal价格](https://modal.com/pricing)：Starter每月$30额度；CPU每物理核秒$0.0000131、内存每GiB秒$0.00000222；A10/L4/T4每秒$0.000306/$0.000222/$0.000164。仅按1000次×2秒、1物理核+2GiB计算活跃推理约$0.035/月，2核约$0.061；这不是账单预测，尚有加载、保留容器、预热、共享账户消耗等。免费指现有月额度可能覆盖，而非永久免费CPU。CPU核单位按[官方资源文档](https://modal.com/docs/guide/resources)的物理核，不能写成vCPU。

不把重推理放进免费Cloudflare Worker：[免费CPU时限10ms、内存128MB](https://developers.cloudflare.com/workers/platform/limits/)不适合本流水线。Worker继续只负责静态页面和轻量代理。量化依据：[ORT量化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)、[FP16限制](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)、[WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)。

## 生成、提交与监看

```bash
python -m infra.windows.prepare_portable_suite_v9 --out artifacts/runs/prepared-v9-portable-20261106-r2 --bundle-out artifacts/ready-suites/code-v9-portable-20261106-r2.zip
```

生成器不联网、不训练。小代码包与签名权重分别传输，解压校验SHA到全新目录 `D:/mahjong/code-v9-portable-20261106-r2`；禁止覆盖历史快照。先运行prepared-suites目录下的prepare.cmd/check.cmd；只有submit.cmd会提交。GPU默认未知进程不放行；开机时重取完整路径，不杀桌面程序。后台WMI进程与聊天/SSH独立，单组崩溃继续下一组；全局截止后不另开窗口。

```bash
ssh win "D:/mahjong/code-v9-portable-20261106-r2/prepared-suites/prepared-v9-portable-20261106-r2/watch.cmd"
```

监看显示队列状态、loss/epoch/mAP与日志尾部；Ctrl+C只退出监看。实际提交命令及精确起止时间见本轮运行回执，不能把准备成功当成训练已启动。
