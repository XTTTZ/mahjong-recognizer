# 冻结验证集上的 GPU 推理对照

`benchmark_runtime.py` 只做推理评估。默认执行本机 CPU 合同、文件 SHA256、图片尺寸/EXIF 预检，不导入 Modal/PyTorch、不执行模型前向。只有显式 `--run` 才创建一次临时 Modal GPU 任务；不部署或修改线上服务，也不训练模型。

三条路径共用同一冻结模型、同一 GPU、相同 CPU 线程数与预热/重复次数：原始 CPU 逐牌正规化 + BF16、批量 GPU float32 正规化 + BF16、可选相同批量路径 + TensorRT FP16。PIL 裁图/补边保持原实现，TensorRT 不做 INT8 或数据校准。每图每次重复轮换路径顺序，计时前后同步 GPU。

## 预检与运行

使用安装了 Modal 1.6.2 的 Python。当前工程的 `infra/modal/cli.py` 文档说明了 Codex bundled Python 和已有认证/代理配置。以下 `python` 应替换为该环境解释器；不需要重新登录。

```sh
python infra/modal/benchmark_runtime.py \
  --model-dir artifacts/models/releases/mahjong-cascade-v26-effv2s-v4-20261027 \
  --valid-dir artifacts/data/roboflow/mahjong-lfhxf-v2/prepared-v4-main/valid \
  --annotation-sha256 42758d7a3a7bec0a4f6fd415c0de5b5a76482a33de1ce45d91c37df4be1abc30 \
  --gpu L4 \
  --out artifacts/runs/runtime-benchmark/validation-l4.json
```

上面命令仅预检，不创建输出报告或 GPU。2026-10-29 本机完整预检通过：150 张、115 个 group、1,396 张牌，图片树 SHA256 `767aa666bcc6b2ae93929f682704b188fe06828c556f111752862dbd2df1dc70`，上传输入共 103,442,253 字节。

在同一命令后添加 `--run` 即执行两条 PyTorch 路径；再添加 `--with-tensorrt` 则构建/复用对应 GPU 的 TensorRT 镜像并比较第三条路径。`--gpu A10` 可以替换 L4，必须在同一种 GPU 内比较，不接受多 GPU fallback 列表。TensorRT 编译和模型加载另记，不混入 forward 延迟。工具使用独立的临时 `mahjong-runtime-benchmark` App 名称，未设置生产部署。

已有完成构建的镜像时，再添加 `--image-id im-实际镜像ID`。该入口直接调用 `modal.Image.from_id()`，使用独立临时 App，不导入 `infra.modal.app`、不重新遍历 ML 镜像构建图，因此无关学生训练代码变化不会触发 TensorRT 重编译。只增加本工具的小文件层；不用重新上传模型权重。未提供 ID 时保留原构建流程。默认 CPU 预检仅检查 ID 格式，不访问 Modal，也不能证明镜像内容/权限可用。

远端注册的是可导入的 `infra.modal.benchmark_runtime.benchmark_gpu` 模块函数，不序列化本机 `__main__` 函数；因此本机 Python 3.12 可以调用现有 Python 3.11 镜像，无需重编译 TensorRT 或调整镜像解释器版本。

实际远端在模型前向前逐文件验证 `/model`，要求模型 manifest 字节 SHA、解析后的完整 manifest 和本机一致。TensorRT 还要求 engine receipt 的模型版本、两个权重 SHA、精度 FP16、GPU 家族、compute capability、两个 engine 文件 SHA 全部匹配。`NVIDIA A10` 与 `NVIDIA A10G` 跨型号复用仅允许 receipt 明确声明 `hardwareCompatibility: AMPERE_PLUS` 的新构建，且必须同为 CC8.6；旧收据仅允许 GPU 完整名称和 CC 均完全相同的评估，不把两个型号仅当名称别名忽略 SM 数量区别。不泛化到其他同 CC 的显卡，也不跨 L4/A10。失败会列明不匹配字段、实际值与期望值，便于区分名称、CC、兼容模式、模型与字节问题。准入不证明引擎可用，最终还须 TensorRT 实际加载和完整逐图 gate 通过。不兼容的预构建镜像不能产生上线通过结果。报告保留实际 GPU 原名、构建 receipt、预构建 image ID 与实际 runtime 源码 SHA。

远端返回前把报告通过 JSON 严格往返转换成原生 `dict/list/str/int/float/bool/null`，拒绝 NaN/Infinity，并显式把 PyTorch 版本转成普通字符串。本机读取报告无需安装 PyTorch，避免版本字符串子类在 Modal 返回反序列化时触发额外模块导入。

默认每图重复 3 次，额外在首张验证照片预热 2 次，每条路径先做同样的模型 synthetic warmup。`--repeats 1..5`、`--warmups 1..3` 有上限；单函数最多 1 个容器、执行超时 3,600 秒。若只需排错可添加 `--limit 3`，但部分验证永远不能通过上线 gate。输出必须使用新文件名以保留已有证据。

## 指标与通过条件

阈值只接受冻结 `score > 0.35`、`acceptThreshold = 0.7`，不提供扫描参数。直接使用定位器预测框和现有 crop context，不使用 GT crop。采用共享 API 正规化、bbox 保留两位、分数保留五位、原始分数决定 uncertainty，IoU 0.5 最大二分匹配避免相邻重复牌被贪心误配。

报告包括全体图片、单牌、10 张以上、原生 10 张以上、oracle ROI 和 13/14 张子组，所有失败/拒绝都在分母。逐图和汇总包含自动整手正确、模型整手正确、牌数量、类别/IoU 命中、拒绝、正确但被拒绝、错误却被接受、置信度/类别/数量/bbox 漂移、重复稳定性，以及 p50/p95 和加速倍数。原始融合分数漂移与 API 舍入后分数漂移分开保留。

`promotionGate.passed` 要求完整验证覆盖、保住每一张 baseline 自动/模型全对图、不增加错误接受、无候选推理失败、重复间类别/数量/接受或正确性稳定。这个逐图 gate 比净整手准确率无下降更严格。原生 10 张以上是主要对照组；本工具不额外打开 train 或 sealed test。完整逐图记录建议保留在忽略 Git 的 `artifacts/runs/`。

延迟范围是同步 GPU 的定位器、PIL 裁图与分类器；不包含图片解码、输出正规化、上传、网络、容器冷启动或模型加载。网页会先把长边限制为 2560、JPEG 质量 .92；本工具使用冻结 native 图片，因此结果不等于完整网页预处理后准确率。网页一致性仍应单独做 valid-only 验收，不重开 sealed test。

## 本机验证

```sh
python -m unittest tests.ml.test_runtime_benchmark -v
python -m py_compile infra/modal/benchmark_runtime.py
```

CPU 测试覆盖阈值边界/舍入、失败拒绝分母、原生/ROI 分组、重复牌二分匹配、class/count/score 漂移、净值相等仍不可互换错图、部分覆盖/重复不稳不得上线，以及 test 路径、路径逃逸、标注/图片 hash 失配。上述准备不证明远端 GPU 比较已通过；只有新生成的完整 benchmark 报告能作为该运行时的上线依据。
