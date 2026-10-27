# V6：有失败证据的两阶段改进，12小时硬预算

**当前是离线准备，未提交训练。Windows关机，V6未完成远端数据准备、包校验或GPU训练验收。** 本机纯CPU测试与冷代码包不能替代这些检查。当前线上研究模型继续使用已冻结的V5发布包。

## 为什么安排这8组

对最终V5的V8两阶段CPU验证胜出配对逐图检查：82张≥10牌照片中，70张自动全部正确，72张模型本身全部正确。12个未自动成功样本中，10个有预测错误，另2个只有置信拒绝。

- 6个错误来自IMG6451–6458同一键盘拍摄条件：真实查看照片后确认键盘按键被当成白板/花牌，侧边较小、倾斜的麻将也有漏检。这些照片属于2个相邻推断group，不能解释为6个独立场景。
- 2张S__散牌图主要出现重复框；最大匹配已能匹配全部真值，不能把多余框误判为分类器普遍不准。
- 其余为密集小牌、画面边缘截断和少量分类混淆。GT crop分类约99.2%，因此先改背景、尺度和裁图覆盖，不增加更大分类器。
- 2张仅拒识的最低融合分为约0.584和0.646。降低接受阈值只会增加覆盖率，还可能接受错误；本轮不将其称为准确率提升，也不依据test调阈值。

逐图诊断属于私有实验记录，不随公开源码分发；上文提供聚合错误类型。该诊断依据完整原图预测与人工查看，未将验证错误照片像素加入训练。随后V26两阶段在完整GPU验证中被选为V5发布模型；V6分类器分支与这个固定定位器配对。

V5实际耗时为：V8单类定位17epoch/36.4分钟，V26单类定位24epoch/70.4分钟，Eff分类19epoch/90分钟后超时。此前RF640约5.7分钟/epoch、YOLO26m约5.9分钟/epoch，旧目录的粗估导致多个模型超时、评估因依赖被跳过。V6按这些全量实测设限。

## 固定的数据和父权重

训练/验证保持冻结V4：train2629图、17868框；valid150图、1396框，其中82张≥10牌；42类映射与source group不变。**以V4 train的标注SHA、真实图片SHA和group为准。** 原始文件`sourceId`可能保留Roboflow导出时的`test/`等前缀，不能仅凭前缀判断当前fold。“本地V2重新分组的holdout”与“Roboflow数据版本2”也不是同一个概念。

背景候选只从当前train内取：先检查与扩张后的真值框零交集，再人工检查全部72份候选的6页contact sheet。拒绝12份含可疑牌面/牌边的候选，接受60份、38个train group，每group最多2份。包含键盘、桌面及鼠标垫；即使某些背面本可作背景，仍保守排除了模糊牌块。它们是已有照片的新增背景视角，**不是60张独立新照片**。review坐标、来源hash、排除理由与contact sheet hash均在[background-v6-review.json](../data/background-v6-review.json)。不依靠“没有标注”直接断言是负样本。

V8分支从相同的V5定位best权重开始；V26两支从相同的V5定位last权重开始；3个分类分支从相同Eff last开始。每支创建新run、重置优化器和scheduler，并记录父run/权重/签名SHA。Eff父训练的超时终态保留，**不补造原run的finishedAt，也不是无损续训**。

## 8个独立训练分支

全部上限12个新epoch，seed2026，lr5e-5。定位器保持名义有效batch32；1280使用batch4而960使用batch8，梯度累积不同，物理batch对BatchNorm可能有影响，因此这是一项显存约束下的实用分辨率对照。

|顺序|分支|唯一变化/比较关系|训练软上限|整组估时/硬超时（分钟）|
|---|---|---|---:|---:|
|1|V8定位960对照|原数据、原光照增强|40|40 /65|
|2|V8定位960+背景|对1仅增加60个已审核背景crop|45|42 /68|
|3|V8定位960+背景+几何|对2仅加轻微几何增强|45|45 /70|
|4|V26定位960+背景+几何|相同数据策略的实用定位器候选|60|50 /85|
|5|V26定位1280+背景+几何|对4提高分辨率，batch4|85|80 /120|
|6|Eff分类320对照|context.08，原光照/框抖动/quarter-turn|80|75 /110|
|7|Eff分类320+相机扰动|对6仅加小倾角及轻微反光|85|80 /115|
|8|Eff分类320+较小上下文|对6仅context.08→.02|80|75 /110|

定位几何增强为±8°、轻微平移/尺度和透视，始终关闭镜像、mosaic和mixup。分类保留quarter-turn；小倾角最多±10°，旋转后扩画布再补边，避免切掉完整牌面；反光是低透明度柔和亮斑，最多22%混合，不使用不透明遮盖关键点数的mask。

目录估时总计**487分钟（约8.1小时）**，再预留60分钟。12小时是上限，缓存提速或提前停止后不为凑时长重复训练。每组包含完整验证，评估另有25分钟上限；各组独立，某组崩溃/超时后继续下一组。

训练软上限只在完整epoch的validation与atomic checkpoint保存后判断。预计下一个epoch放不下时正常结束，run记录`completionReason=budget_finished`、实际与计划epoch数、elapsed、父权重和`convergenceClaim=false`。达到12epoch则记录`epoch_limit_reached`，同样不宣称已收敛。外层队列保留进程硬超时和总截止，异常部分权重不作为训练完成。

每组用实际定位框做全valid两阶段评估，最多2个不同checkpoint候选；定位分支配固定Eff last，分类分支配固定V26 last。使用已有`recover_terminal`的CPU评估与终态证明，避免旧Eff超时再次阻断评估。CPU研究胜出者仍需GPU及网页链路验收，不能自动覆盖线上模型，也不对所有新detector/classifier做无界组合搜索。

## 无损缓存与预处理边界

`prepare_crop_cache_v6`把train/valid原图中的牌面周边像素存成PNG envelope，减少每个crop反复解码整张高像素照片，不增加新数据。schema2保存原global bbox、原图尺寸和整数envelope位置。V6 `EnvelopeImage`先在**原global坐标**执行线上`crop_face`的jitter与取整，再平移已取整坐标；不修改线上默认实现。

回归测试包含真实annotation2641的浮点边界（直接使用cache局部float会偏1px）、边缘牌、context.02/.08/.16和随机jitter。schema1旧缓存明确拒绝，缺global字段也拒绝；不能把旧缓存冒充ready，必须新目录重建。Windows目标尚未构建V6缓存。

离线val使用native原图，网页canvas会先缩到最长边2560并以JPEG质量.92上传。4K原图的分类crop细节可能因此改变，已有87.8%验证/66.0%封存主指标不是整条网页预处理后的准确率。**V6最终候选还需valid-only网页预处理一致性验收**；当前Windows关机，未实测，不为此重新打开sealed test或扩大已选模型评估。

V5 sealed test已用于一次发布审计。本轮不读取、不重新筛选、不根据其结果设计训练样本或阈值；V6最终泛化判断需要独立、已标注的棋牌室手机holdout，不能继续把旧test当反复调参工具。

## 冷准备、检查、提交

默认目标：`D:/mahjong/code-v6-ready-20261028`；独立环境`mahjong-v6`。已有本机exec缓冲确认此前conda clone退出0，但未做V6实时package/pip复核。模板package pins与父资产hash是预期配置，**不是当前Windows验证证据**。

本机生成并验证冷包，不联网、不训练：

```bash
python3 -m infra.windows.prepare_refinement_suite \
  --out artifacts/ready-suites/v6-12h \
  --bundle-out artifacts/ready-suites/code-v6-ready-20261028.zip \
  --gpu-background-paths artifacts/runs/queue-prep/gpu-background-paths-current.json
```

生成器可接收`--parent-assets`、`--expected-environment`、`--v8-parent`、`--v26-parent`、`--classifier-parent`。公开源码中的pins/父hash模板不含私有机器日志，但原始父训练run和完整训练checkpoint没有随源码发布。复现者需准备自己的签名父run与数据；推理release里移除训练状态的classifier不能当exact optimizer resume。默认磁盘/解释器路径按本项目Windows配置，其他机器需调整配置。

Windows上线后，将ZIP与`.verify.py`复制到staging，以冷包receipt中的SHA执行`unpack`到**全新**code目录。包内已含`prepared-suites/v6-12h/`。然后只做CPU准备：

```bash
ssh win "D:/mahjong/code-v6-ready-20261028/prepared-suites/v6-12h/prepare.cmd"
```

这一步不训练、不提交：环境缺失时离线clone；核验冻结源、7份父资产、package pins与pip；生成或严格复核背景数据和schema2 cache；运行全部16个step的参数边界检查及8个训练入口的CPU预检。存在不完整数据目录时保留并报错，不静默覆盖。CPU进程设置`CUDA_VISIBLE_DEVICES=-1`，不改系统环境。

再次只检查、读取进度的命令：

```bash
ssh win "D:/mahjong/code-v6-ready-20261028/prepared-suites/v6-12h/check.cmd"
ssh win "D:/mahjong/code-v6-ready-20261028/prepared-suites/v6-12h/watch.cmd"
```

**本轮不执行提交。** 未来用户明确恢复训练、prepare/check通过后，才手动运行：

```bash
ssh win "D:/mahjong/code-v6-ready-20261028/prepared-suites/v6-12h/submit.cmd"
```

`submit.cmd`是唯一包含`--start`的入口，先重新校验再由WMI独立后台运行；关闭聊天/SSH不影响它。启动时仍按完整exe路径匹配历史已核验的桌面GPU allowlist，Python/ML运行时始终阻塞。该allowlist来自V5证据，不是本次Windows live核验；未知路径会安全等待并明确失败，不能按通用进程名放行。

断电、Windows重启、GPU驱动故障不能由内存队列保证持续运行。中断历史与checkpoint保留，重新启动同配置只继续未处理的独立任务；不会悄悄重训或伪造成功。具体准备状态与冷包hash见`ready-v6.json`，远端检查通过前其状态保持unverified。
