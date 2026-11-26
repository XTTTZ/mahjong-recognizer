# 网站发布与验证

正式网址：https://mahjong.avalonxty.site 。独立 Cloudflare Worker 名称 `mahjong`，前端与 `/api/*` 同域；没有 Neon、对象存储、照片历史或用户账户依赖。Poker 与 Avalon 的 Worker、域名、数据库不属于本项目部署操作。

2026-10-28 已将线上模型更新为 `mahjong-cascade-v26-effv2s-v4-20261027`，原 Modal App `mahjong-inference` 和接口地址保持稳定。Cloudflare 版本 `c0c97a89-8252-4d48-b10d-0afd5d909d8b`，网页版本 `2026-10-28-cascade-trial`。手机网页上传→GPU推理→中文结果→逐牌人工修改已完成两次真实线上验证；320/390/430px 浏览器仿真页面无横向溢出，20项单元/网关测试、12项浏览器测试通过。

首次请求21.527秒，紧接着的请求4.423秒；服务内部处理分别5.982秒、2.947秒。两次均只发一次POST，模型版本校验一致，标注14张的验证照片均返回14张，人工将“四条”改为“一万”成功。此结果证明流程和交互可用，牌数相同不等于整手所有牌面均识别正确，不能当成现场准确率验收。具体耗时受GPU分配、容器启动及照片内容影响。凭证见 `release-20261028.json`，本机截图与报告在 `artifacts/web/live-20261028-cascade-cold/` 和 `artifacts/web/live-20261028-cascade-warm/`。

当前 UI 明确标记“试用模型 · 结果请逐张核对”。网页上线与模型准确率达标是两项验收，仍需独立真实棋牌室照片验证。

2026-10-27 首次联通使用的旧RF-DETR基线在同一14牌照片上只返回3张，已被本次模型替换；历史凭证 `release-20261027.json` 保留。响应的 `quality.status=ok` 只表示已返回牌的分数没有触发低置信规则，不能证明没有漏牌。人工修改明确标记，不能计入模型自动正确率。

页脚可见 [完整源码（AGPL-3.0）](https://github.com/XTTTZ/mahjong-recognizer) 与 [模型下载](https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027)，链接在320px下保留44px触摸区。公开仓库独立于原私有工程，发布更新时需同步对应运行源码及模型来源。

## 首次部署或更新

在项目根目录使用 Node 24/26：

```sh
npm ci
npm run worker:types
npm test
npm run test:e2e
npm run build
npm run models:stage
npm run worker:dry-run
npx wrangler deploy
```

`wrangler.jsonc` 是非秘密配置源：`routes` 的 custom domain 使用 `mahjong.avalonxty.site`，`MODAL_URL` 指向 Modal ASGI 的 `/recognize`，`INFERENCE_ENABLED` 控制实际调用。首次可设 `false` 发布网页；前端通过 `/api/health` 读取配置状态，服务准备期间保留拍照、上传、旋转与裁剪，禁用识别按钮。`health` 不启动 GPU，也不宣称推理服务已探测。

Modal 使用 proxy auth，Worker 通过 `Authorization: Bearer ...` 转发请求。秘密只放在 Cloudflare secret，不写入前端、Git 或 Wrangler vars：

```sh
npx wrangler secret put MODAL_PROXY_TOKEN < .secrets/modal-proxy-token
```

该本地文件由 Modal proxy token 初始化流程创建，已被 `.gitignore` 排除；更换后重新写入 secret。不要把其内容复制到命令参数、日志或文档。使用 `npx wrangler secret list` 只能核验名称。

前端 Static Assets 直接分发，安全响应头在 `apps/web/public/_headers` 设置；不能依赖 Worker 处理静态资源时的代码分支。`/api/*` 在 Worker 内自行设置 `no-store`、`nosniff` 等响应头。静态资源与 API 各自的配置符合 [Cloudflare headers 文档](https://developers.cloudflare.com/workers/static-assets/headers/)。

## 已发布站点验收

只检查主页、HTTPS、安全头、手机布局及上传预览，不触发 GPU：

```sh
node infra/web/verify-live.mjs --image /absolute/path/to/validation-photo.jpg
```

实际调用一次线上 GPU，验证手机上传、自动结果、坐标显示与逐牌人工修改：

```sh
node infra/web/verify-live.mjs --image /absolute/path/to/validation-photo.jpg --recognize
```

使用 `--expect-model mahjong-cascade-v26-effv2s-v4-20261027` 可防止误把旧模型作为本次验收。`--out` 选择新目录，保留历史报告。`--proxy http://127.0.0.1:PORT` 仅在已有、受信的临时代理通路上使用，不关闭TLS验证、不修改系统网络；本次最终验收直接使用本机网络，未依赖Windows隧道。

报告和 320/390/430px 截图保存在 `artifacts/web/live/`。这项测试只验证真实网络链路，图片必须来自允许使用的验证集；不能据此声称整手识别准确率。冻结的 test 集留给最终模型验收。真实手机相机、Safari/Chrome、HEIC 解码与棋牌室照片仍需设备验证。

网关拒绝跨站 Origin、非 multipart 请求和超过上限的上传；对个人/站点限流，90 秒上游超时，客户端等待95秒，无自动重试。照片在上传时流式转发到 Modal，不写日志、数据库或对象存储。服务返回不确定牌提示，人工修改只保留本次页面内存。

Workers 出站请求固定使用 `redirect:'manual'` 并拒绝任何3xx，避免将认证或照片转发到其他地址。[Workers Request 文档](https://developers.cloudflare.com/workers/runtime-apis/request/)明确只实现 `follow/manual`，`redirect:'error'` 会在请求构造时抛错。此边界已有回归测试，并通过真实云端调用验证。运行时日志只记录请求编号、字节数、耗时、状态、阶段及有限的传输错误；不会记录照片、上游响应内容或认证头。

## 统计脚本与控制台诊断

已在无扩展的Chromium复现 Cloudflare 自动注入的 `static.cloudflareinsights.com/beacon.min.js/...` 被 `script-src 'self'` 阻止。这条来自站点的自动统计注入。现在通过本应用的 `Cache-Control: public, max-age=0, must-revalidate, no-transform` 停止注入，按 [Cloudflare官方说明](https://developers.cloudflare.com/web-analytics/get-started/)处理；没有放宽任意第三方脚本，也没有改动共享域名其他应用的统计配置。

最终两次线上验收均无外部脚本、CSP违规、console.error或未捕获异常，公开源码与模型入口存在。`LanguageDetector`、`ImmersiveTranslate` 未在干净浏览器复现，本站源码没有调用这些API；此类浏览器/翻译扩展日志不能单独说明GPU是否正常。

GPU故障另有真实证据：旧站请求 `e6065850-0161-44ce-9474-b5359164825f` 返回504，发生在L4分配排队阶段，与统计脚本无关。新Modal部署允许L4/A10回退，函数执行上限90秒，min0/max1、空闲15秒回收；网关90秒与客户端95秒配套，前端不自动重发请求。最终冷/热请求均200，配置检查 `/api/health` 仍不启动GPU。

## 故障与回滚

### 2026-10-30：可选本地性能测试

`/benchmark`独立加载浏览器推理；首页仍使用既有Modal服务。用户主动加载后才下载约119 MB ONNX模型和25.5 MB运行库。照片不发到服务器，取消/离开页面会结束专属Worker。普通首页没有模型预下载。使用说明和实际浏览器复测命令见[本地测试说明](../../apps/web/src/benchmark/README.md)。

Worker只代理`/models/portable-v1/`下三个固定公开文件，以流式传输避免将权重放入Worker内存。发布源固定为公开GitHub release `runtime-onnx-fp32-20261030`，不接收任意URL，不转发用户Cookie或认证。模型按manifest中的大小和SHA-256在浏览器校验。ONNX先做训练路径脱敏，计算图和张量保持不变。

只有测试文档和`/assets/local.worker-*.js`添加`wasm-unsafe-eval`与同源隔离头；首页仍是`script-src 'self'`。文档与模块Worker都需要COEP，否则浏览器会在模型请求前阻止Worker。`run_worker_first`覆盖这些路由，避免静态资源绕过必要的响应头。运行库为固定npm版本的本地静态文件，不允许CDN脚本或远程照片上传。

需要停止实验入口时，移除首页性能测试链接并保留云端默认识别即可；不要更改当前GPU权重。若模型文件升级，创建新版本路径与release，勿覆盖已声明immutable的文件。

先访问 `/api/health`：`configured=false` 表示 URL、开关或 secret 未齐全；`configured=true` 只表示配置齐全。随后检查 `npx wrangler tail mahjong` 和 Modal App 日志，日志不得记录照片或凭据。

`INFERENCE_ENABLED=false` 后重新部署可暂停 GPU 调用并继续提供前端工具。查看 `npx wrangler versions list`，使用 `npx wrangler rollback <VERSION_ID>` 可回滚到已知版本；注意同时核对回滚版本所绑定的 Modal URL 与模型版本，secret 不随前端源代码回滚。

## 2026-11-25 首页识别延迟

首页默认快速识别。页面可见时并行检查配置并预热 CPU，用户拍照和核对取景时容器可保持 120 秒。精细识别使用 GPU。本机模式由用户启用，下载及校验在后台进行，回访时通过 Cache Storage 复用已验证的模型。普通云端访问下载模型 0 字节。存储被浏览器回收时会重新准备。

Cloudflare 版本 `e9c1feb1-2c82-42e0-9482-52f846f7c244`。M4 Pro 桌面 Chrome 的生产站测试用同一张已知 14 牌图。后台准备期间提交云端快速识别，点击到结果渲染 4.36 秒，网关内 2.85 秒，CPU 服务处理 0.83 秒，其中推理 0.44 秒。各计时范围嵌套。首次本机准备 30.66 秒，实际传输 32.59 MB。回访刷新到就绪 1.22 秒，权重和 WASM 传输 0 字节。本机三次识别 0.83–0.88 秒，三次牌面均正确。该设备和网络的结果不能推算手机或其他运营商的延迟。

快速识别对长边超过 1600 像素的照片另外生成 JPEG（长边 1600、质量 0.83），在字节数更少时使用。精细识别、预览和本机版继续使用最长边 2560、质量 0.94。验证时在 Chromium 从相同原图生成两个版本，使用上线 V10 的 CPU ONNX、同一接受阈值和原始坐标系评估。冻结验证集中的 60 张大图上传总量从 83.96 MB 降到 21.50 MB。150 张自动整手正确从 135 到 136，至少 10 张牌的照片从 73/82 到 74/82，13–14 张牌保持 44/45。逐图检查中没有原本正确而压缩后错误的照片，误接受仍为 1/150。这次评估使用浏览器重新编码的验证照片。

## 2026-11-26 Windows 首次加载

Windows Edge 158 从全新的浏览器配置访问生产站。初次开启时，系统代理指向无人监听的本机端口，浏览器在发出 HTTP 请求前返回 ERR_PROXY_CONNECTION_FAILED。稍后观察到系统代理已关闭，默认配置打开完整首页用时 2.10 秒。同期五次独立 HTTP 客户端请求首页均成功，中位 0.65 秒，Cloudflare 均由 AMS 节点响应。

在系统代理关闭后的直连测试中，旧版首次准备轻量模型耗时 8.08 秒，模型与运行库实际传输 32.59 MB。定位权重 2.94 秒，WASM 1.61 秒，分类权重 1.51 秒。浏览器刷新仅需 1.12 秒就绪，权重与 WASM 的传输均为 0 字节。同一张已标注的 14 牌照片，三次本机点击到渲染耗时 0.88–0.89 秒，三次均整手正确。完成页面预热后点击快速识别耗时 2.14 秒，其中 Modal CPU 模型推理 0.39 秒。

同一台 Windows 主机对相同生产模型文件做四轮交叉测速。每轮完整读取并核对 29,853,745 字节，两次串行耗时 5.21 和 2.96 秒，两次并行耗时 2.21 和 1.70 秒。关闭浏览器 HTTP 缓存，CDN 缓存状态有变化。这项测量只覆盖两个权重的网络传输。

版本 `8f1c6f57-9d76-4ef7-8feb-59284baed963` 在用户允许本机准备后同时读取两个轻量权重，校验完整性后依次初始化 ONNX 会话。总进度按字节计算，同时下载的耗时按真实经过的时间统计。已缓存的权重可跨版本复用。119 MB 对照配置维持逐个文件读取和初始化以控制内存。
