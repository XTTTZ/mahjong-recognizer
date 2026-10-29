# 冷启动与两阶段推理优化

本轮保持已公开的 YOLO26s→EfficientNetV2-S 权重、检测960、分类320、8%上下文和固定阈值 .35/.7，优先验证整手准确率。模型版本仍为 `mahjong-cascade-v26-effv2s-v4-20261027`，运行后端和模型权重版本分别记录。

## 预热与GPU快照

网页第一次可见且在线时，独立 `POST /api/warmup`，Worker带服务端凭据只访问模型 `/health`；无需先上传图片。前端60秒去重，网关每IP每分钟1次、全站每分钟6次，和真实识别限流分开。限流返回明确的`deferred`，预热失败不会禁止选图或触发自动识别。没有定时保活。容器最少0、最多1，空闲30秒后可缩零，不能保证任意时刻都有GPU可用。

Modal以 `enable_memory_snapshot=True`、`experimental_options={"enable_gpu_snapshot":True}`、`@modal.enter(snap=True)` 加载模型和预热检测器及分类batch 1/14/32，然后抓取GPU内存。库初始化和首次算子计算进入快照。GPU快照仍为官方Alpha功能；改镜像/代码/GPU后会重新创建，不能把初次建快照耗时当作恢复耗时。参见[官方说明](https://modal.com/docs/guide/memory-snapshots)。

需要禁用快照时显式设置 `MAHJONG_SNAPSHOT=0` 再部署；不让异常时自动重发上传请求。回退前先在候选服务验证，不改公开上传协议。

## 裁剪、规范化与Batch

- 保留训练时PIL bilinear、原宽高比、114灰色填充与边缘裁剪；没有用ROIAlign或GPU双线性擅自替代像素规则。
- 同一照片最多200个检测，按32个牌面分组，保留余数；不等待其他用户照片来凑batch。
- 一组裁剪先合并为连续uint8数组，一次传GPU，再整批float32除255、ImageNet mean/std规范化；均值方差只分配一次。原`image_tensor`保留用于训练和原始对照。
- CPU线程匹配分配的2核，避免在低CPU容器中大量线程争抢。
- 单个TensorRT执行context串行使用，HTTP层加锁；不同照片不会同时覆写context绑定。结果仍按真实检测框逐一融合，分数为detector×classifier。
- API补充decode/inference/postprocess分段时间；网络与冷启动时间另测。

## TensorRT FP16

`MAHJONG_RUNTIME=tensorrt MAHJONG_GPU=A10`（或单L4）在同型号GPU上编译两个引擎。Torch2.8.0/CUDA12.8、TensorRT10.13.3.9、ONNX1.19.0；检测器固定batch1/960，分类器动态batch1–32、最优14/320。工作区每个引擎2GiB。构建无校准数据、不使用train/valid/test像素；INT8另作实验，不直接用于线上。

`/engines/runtime.json`记录两份原权重和两份引擎SHA、GPU/SM/计算能力、依赖版本、精度与batch范围。两引擎都真实启用`AMPERE_PLUS`并读取序列化引擎属性复核，允许A10资源池中的A10G80SM与A1072SM；旧固定型号引擎已弃用。加载时还核验IO contract/metadata/hash；其他GPU需先重新验证。原始PT权重仍是可复现源。

```sh
MAHJONG_MODEL_DIR=/absolute/path/to/frozen-model \
MAHJONG_MODAL_APP=mahjong-inference-candidate \
MAHJONG_RUNTIME=tensorrt MAHJONG_GPU=A10 \
"$CODEX_PYTHON" infra/modal/cli.py deploy infra/modal/app.py
```

本机`CODEX_PYTHON`由工作区依赖工具确定。生成的镜像确认无误后，可设 `MAHJONG_PREBUILT_IMAGE_ID=im-...` 复用已验证的不可变镜像，省去重新编译。部署器仍校验本地权重，并在容器初始化时比对预期模型版本和两个权重hash；错误镜像拒绝服务。其他账户先自行构建，不能假设原账户镜像ID可用。

模型文件和源代码先冻结再构建，避免长时间引擎编译期间工作区其他文件改动导致上传失败。镜像内没有上传照片或服务凭据。

## 验证与发布

[对照工具](benchmark_runtime.md)比较同一GPU的原BF16、批处理BF16及FP16 TensorRT。固定完整valid150张/1396个tile，主指标是原生照片至少10牌的82张整手完全正确；另报告全150、ROI、tile precision/recall、逐图退化、误接受、输出漂移与p50/p95。只用valid，不重复开启封存test。完整比较通过后再更新稳定Modal App，Worker域名和保护凭据保持不变。

冷启动必须确认容器已归零或只终止自己的候选空闲容器，再请求并查不同容器ID及`Restoring Function from memory snapshot.`；仅等待40秒但仍是同容器的测量只能算热请求。`probe_runtime.py`保存HTTP时间和服务内分段，不输出token或原图片。

真实网站检查：

```sh
node infra/web/verify-live.mjs --recognize --image /path/to/authorized-validation.jpg \
    --expect-model mahjong-cascade-v26-effv2s-v4-20261027 --expect-backend tensorrt-fp16 --out artifacts/web/fresh-live-run
```

不加`--recognize/--warmup`的检查会拦截预热并标明只读；模拟识别的`verify-scoring-live.mjs`验证计倍交互，不证明模型准确率。

## 预算

按2026-10-29[官方价格](https://modal.com/pricing)，A10约$1.10/GPU小时、2核CPU约$0.094/小时、4GiB内存约$0.032/小时。若每月1000次相互独立的访问，每次从启动至缩零总占用40秒，约$13.6/月；100秒约$34.1/月。实际页面访问次数（即使未识别）、排队/快照、重复部署和同账户其他作业都会影响用量，不能按“1000次识别”保证总费用。Starter当前$30/月额度；没有升级订阅或修改消费限额。
