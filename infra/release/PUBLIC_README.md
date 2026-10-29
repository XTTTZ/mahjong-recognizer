# Mahjong Recognizer / 麻将识别

[手机网站](https://mahjong.avalonxty.site) · [模型与校验值](https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027)

独立的手机拍照识别应用：React 前端 → Cloudflare Worker 上传网关 → Modal L4 GPU。照片只在请求期间处理，不持久保存，也不会自动回流训练集。不需要数据库。

当前模型先用 YOLO26s 检测牌面，再用 EfficientNet V2 S 分类。共42种正面牌；支持拍照、选图、旋转、区域裁剪、识别框展示、逐牌更正、补牌、删除和撤销。识别后按广东大牌136张规则判断胡牌与倍数，特殊情况由用户确认，编辑后在本机重算。**实验版，结果必须逐张核对。**

计倍采用用户确认的“兼容牌型相加，按表排除重复项”；碰杠需声明已有实体牌，3倍起胡，中码最后乘算。照片不能确定海捞、杠开、门清等历史条件，需手动选择。来源转写与未明事项见 [规则解释](docs/rules/source-interpretation.md) 和 [实现口径](docs/rules/IMPLEMENTATION.md)，原始私人规则图片不随工程分发。

GPU运行时可采用两阶段TensorRT FP16及Modal GPU Memory Snapshot；PIL裁剪几何保持训练规则，牌面整批传GPU规范化。完整valid运行时对照保持主指标72/82，内部p50约58→30ms；候选真实快照恢复约4.75秒，后续一次识别约1.16秒。样本与测量范围见[验证报告](docs/models/RUNTIME_20261029.md)，构建、上线和回退见[操作说明](infra/modal/OPTIMIZATION.md)。

验证集至少10张牌的82张照片中，自动整手完全正确72张（87.8%）。仅在验证集锁定模型与阈值后进行一次封存测试：47张多牌照片中自动整手完全正确31张（66.0%），忽略置信度拒绝时模型全对35张（74.5%）。这些数据来自同一个公开来源的分组留出，不能保证陌生棋牌室的准确率。单张分类准确率不能替代整手识别率。

## 本地运行

使用 Node.js 24 或26：

```sh
npm ci
npm run build
npm test
npm run test:e2e
npm run worker:dev
```

`npm run dev` 只启动前端。真实识别需要配置后端；模拟接口的浏览器测试不证明云端模型可用。

## 部署

1. 下载模型 release 的 `model.json`、`detector.pt` 和 `classifier.pt` 到同一目录，校验 `SHA256SUMS`。
2. 安装 Modal SDK 并登录；将 `MAHJONG_MODEL_DIR` 指向该目录。按 [Modal 操作说明](infra/modal/README.md) 部署保护端点。默认优先L4、可回退A10，空闲缩零、最多一个容器。
3. 在自己的 Cloudflare 账户配置 `wrangler.jsonc` 的路由、账户、Modal URL 和限流命名空间。示例域名属于原项目，部署自己的实例时必须替换。
4. 以 secret 保存 `MODAL_PROXY_TOKEN`，使用 Modal 的 proxy token，不能使用管理凭据。参见 [Worker 源码](apps/worker/index.ts) 和 `.dev.vars.example`。
5. 运行 `npm run worker:deploy`，用真实照片验证完整手机流程。`/api/health` 只确认配置；首次可见页面会独立调用限频的 `/api/warmup`，提前加载模型，失败不会阻塞识别。

部署和训练需要用户自己的账户及 GPU 资源；空闲缩零可以降低支出，不代表供应商保证永久免费。网页不内嵌任何管理凭据。

## 数据和可复现训练

数据来源为 [Roboflow Universe Mahjong v2](https://universe.roboflow.com/mahjong-anaun/mahjong-lfhxf/dataset/2)，许可 CC BY 4.0。原始快照11921张；清洗、去重和分组后的主训练集2629张、验证150张、封存测试121张。额外4671张的扩展训练版本另有清单。项目不分发原始照片。

参见 [数据卡](docs/data/DATASET_CARD.md)、[V4 划分与复现](docs/data/DATASET_V4.md)、[数据工具](ml/data/README.md)、[训练与评估](docs/models/YOLO_CASCADE.md)、[下一轮V6方案](docs/models/V6_REFINEMENT.md) 和 [Windows 后台队列](infra/windows/EXPERIMENT_QUEUE.md)。`ml/train`、`ml/eval` 和 `infra/windows` 包含训练、终止恢复、整手评估、模型导出及独立于 SSH 会话的串行执行器。环境和预训练下载固定版本与校验值；本机路径、账户和计划哈希需重新生成。V6使用历史签名父权重；其他安装应先运行基础训练生成自己的父run及receipt，不能绕过这些校验或把公开推理权重当作精确resume。

训练与选阈值只使用 train/valid。封存测试已有使用记录，后续改进不得反复按其结果调参；应另采集独立手机场景作为新评测集。训练默认禁止镜像麻将文字。下一轮实验代码的存在不会自动启动训练，提交队列必须是用户明确发起的操作。

[V7效率实验](docs/models/V7_EFFICIENCY.md)准备了12小时硬截止的11组队列：保留数据／检测器／分类器对照，增加MobileNetV3-Large监督与EffV2S教师蒸馏，以及仅train校准的CPU INT8对照。代码在本机准备不等于Windows已经就绪，开机同步后必须重新做路径、环境、父权重及数据签名预检。队列失败会转下一独立组，依赖失败的量化组跳过；没有从网站或部署过程自动训练。

## 许可

完整应用源码及本次选中权重以 **AGPL-3.0-only** 发布。通过网络提供修改版本服务时，遵循许可证的对应源码提供要求。上游及数据出处见 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。其他第三方依赖保留其各自许可。本仓库是独立麻将工程的干净快照，不包含其他应用、私有仓库历史、凭据、原始照片或私有实验日志。
