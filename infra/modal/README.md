# Modal 管理环境与连接排障

2026-10-25验证：SDK1.6.2，profile/workspace `MODAL_WORKSPACE`，当前environment为`main`。管理CLI安装在Codex bundled Python的user site，不将认证值写入工程。

## 本机连接故障与修复

表层报错`Could not connect to the Modal server`，最内层实际是`python-socks`缺失。Modal1.6.2通过`urllib.request.getproxies()`读取代理，macOS系统代理也会被读取；因此即使`HTTP_PROXY`等环境变量为空，本机仍使用已有`http://127.0.0.1:7897`。强制IPv4或重新登录不能修复缺失依赖。

已使用经过验证的Codex bundled Python执行官方可选依赖安装：

```sh
"$CODEX_PYTHON" -m pip install --user 'modal[api-proxy-support]==1.6.2'
```

`CODEX_PYTHON`需先从Codex的workspace dependencies工具获取，不能把变量占位值直接当路径。此次新增`python-socks==3.1.1`、`aiohttp-socks==0.12.0`。没有更改token、系统代理、全局网络、TLS验证或账户预算。

默认管理命令已恢复，继续使用原有系统代理：

```sh
"$CODEX_PYTHON" infra/modal/cli.py app list --json
"$CODEX_PYTHON" infra/modal/cli.py environment list --json
```

如未来代理临时不可用，已验证可使用官方配置仅对当前进程直连，不修改shell或系统配置：

```sh
"$CODEX_PYTHON" infra/modal/cli.py --direct app list --json
```

该helper只设置`MODAL_DISABLE_API_PROXY=1`，不会禁用TLS、改DNS、修改SDK或读取/输出认证值。直连取决于当时网络可达性，默认仍建议保持现有代理。

## 本次只读验证结果

| 查询 | 结果 |
| --- | --- |
| `app list --json` | 成功，空列表 |
| `environment list --json` | 成功，仅`main`，active=true |
| `workspace settings list --json` | 成功，default-environment=main，image-builder-version=2025.06 |
| `billing summary --json` | 成功，当前月metered_cost=0、billed_cost=0 |

以上查询均通过已修复的默认代理路径完成。这里的billing summary是使用量/账单快照，不证明GPU配额、未来免费额度或费用封顶设置。费用封顶以主任务另行确认的账户控制台状态为准。

本次没有启动云计算、部署App、创建Volume/Secret、修改environment或更改预算。最终网站域名由主任务设置为`mahjong.avalonxty.site`；管理CLI连接成功不等同该网站已能识别图片。

## 模型发布与接口

部署器会先验证冻结的`model.json`、类别与checkpoint hash，镜像只包含推理代码和冻结模型。当前RF/RT及YOLO/两阶段adapter共用HTTP协议；新候选只有训练和完整验证后才能打包。YOLO路线目前仅供实验准备，线上试用仍使用RF-DETR。

```sh
MAHJONG_MODEL_DIR=/absolute/path/to/frozen-model \
MAHJONG_MODAL_APP=mahjong-inference \
"$CODEX_PYTHON" infra/modal/cli.py deploy infra/modal/app.py
```

若本机代理上传出现`Connection lost`，可仅对此次进程增加`--direct`，保持系统配置、登录和TLS校验不变。SDK的API代理与大文件上传都可能受代理连通性影响，重试前先确认现有App状态。

模型大文件在本地按4MiB切片以并行上传，镜像构建时逐片校验SHA256后原样重组并验证整文件hash；没有转半精度、重新量化或修改checkpoint数值。切片缓存位于忽略Git的`artifacts/models/modal-upload/`。模型像素处理和阈值均来自冻结manifest，不在部署时另行猜测。

GPU配置为L4、最少0容器、最多1、空闲15秒缩容，请求上限60秒。首次请求需要加载模型，热请求耗时与首次冷启动不能混为一谈。`/health`在模型容器上会启动GPU；网站的`/api/health`只确认网关配置，不调用Modal。照片只用于单次推理，不写数据库/对象存储。

Proxy token由`modal.Workspace.from_context().proxy_tokens.create(...)`或官方CLI创建，combined token只存本机`.secrets/modal-proxy-token`和Cloudflare secret `MODAL_PROXY_TOKEN`，不进入镜像或前端。直接无凭据调用Modal应返回401；只有Worker持有转发凭据。参见[Modal官方认证说明](https://modal.com/docs/guide/webhook-proxy-auth)。

上线真实浏览器验收使用`infra/web/verify-live.mjs --recognize`；应记录模型版本、一次调用的结果与耗时。它验证网络链路，不能替代整手准确率评估。已有v3/960模型的3张多牌验证图整手全对数为0，网站明确标记试用并要求逐牌核对，`productionValidated=false`。
