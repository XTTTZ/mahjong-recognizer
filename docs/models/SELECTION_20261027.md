# 2026-10-27 当前模型选择与发布包

当前选择是 **YOLO26s单类牌面定位 → EfficientNetV2-S 42类分类**，版本`mahjong-cascade-v26-effv2s-v4-20261027`。它是现有权重在验证集上的最佳候选，仍属于需要逐牌核对的研究版本。封存测试显示真实泛化仍有明显缺口，不能把验证集成绩当作棋牌室准确率承诺。

## 验证后选择，封存测试不调参

先完成V26定位器best/last与EfficientNet best/last的4组CPU全验证，再对V26和V8两阶段的胜出配对做完整Windows GPU复核。RF640 EMA/last也完成了GPU补评估。全部使用相同v4 train/valid字节与42类映射，不用真值框裁图替代定位器。

主指标为82张原始照片中≥10张牌的完整识别；框、类别、数量必须全部正确，低置信拒绝和异常仍留在分母。匹配使用同类IoU≥0.5最大匹配；不强制补齐14张，也不按每类4张修正预测。

| 现有候选 | 设备与输入 | 自动整手正确 | 模型本身全部正确 | 说明 |
| --- | --- | --- | --- | --- |
| YOLOv8s直接42类 | GPU，960 | 38/82，46.34% | 44/82，53.66% | 完成训练后的best/last选择 |
| YOLO26s直接42类 | GPU，640 | 22/82，26.83% | 32/82，39.02% | 输入与规模不同，不能单独归因架构 |
| YOLO26m直接42类 | CPU，960 | 48/82，58.54% | 53/82，64.63% | 190分钟超时权重补评估 |
| RF-DETR Small control | GPU，640 | 9/82，10.98% | 10/82，12.20% | 165分钟超时，EMA/last补评估 |
| YOLOv8s定位→EffV2-S | GPU，960/320 | 71/82，86.59% | 73/82，89.02% | CPU为70/82，GPU数值差异影响1张 |
| **YOLO26s定位→EffV2-S** | **GPU，960/320** | **72/82，87.80%** | **74/82，90.24%** | CPU/GPU这组统计一致，作为当前winner |

候选按自动整手正确率、模型全部正确率、接受覆盖率依次排序。V26与V8只有1张GPU验证照片的差距，不能解释为具有统计把握的架构优势。验证数据来自拼合的公开资料，source group仍含推断，阈值与候选都在同一验证集选择，数字可能乐观。

进一步逐图比较了V26的150张CPU/GPU输出：每张的全部正确、自动成功、接受/拒识与异常状态相同；8张本来已出错的照片在保留类别集合上有低精度差异，其中5张预测数量不同。因此这里只确认逐图正确/拒识结论一致，不宣称原始预测位级相同。发布包及封存测试均使用GPU推理路径。

之后把winner、权重SHA、模型manifest、验证报告、阈值和数据元信息写入不可覆盖的`winner-lock.json`，才打开封存test。唯一一次打开记录为`SEALED-TEST-OPENED.json`。不根据test换模型、调阈值或重新筛图。

## 冻结工作点与封存结果

模型使用定位器`last.pt`及classifier`last.pt`；检测输入960、分类输入320、裁图上下文8%，保持整个牌面比例并补边。融合分数为检测分数乘分类softmax；保留分数严格大于0.35的结果，低于0.7标为不确定。以上在test之前已固定。

封存test为121张、916个牌面、112个source group，没有oracle ROI。以下保留固定工作点，异常数为0：

| 范围 | 自动全部正确 | 模型全部正确 | 接受结果正确率 |
| --- | --- | --- | --- |
| **原图≥10张牌，主指标** | **31/47，65.96%** | **35/47，74.47%** | **31/35，88.57%** |
| 13或14张牌子组 | 19/22，86.36% | 19/22，86.36% | 19/21，90.48% |
| 全部照片，含单牌 | 101/121，83.47% | 105/121，86.78% | 101/106，95.28% |

不能用含46张单牌的全图结果，或只有22张的13/14牌子组，替代47张主指标。47张中仍有4张错误结果未被置信策略拒绝，因此用户始终需要核对牌面。当前不足以支持“自动识别准确率达标”的结论，也缺少独立棋牌室手机拍摄测试集。

## 完整性与部署接口

发布包本机路径：`artifacts/models/releases/mahjong-cascade-v26-effv2s-v4-20261027/`。

Windows路径：`D:/mahjong/packages/mahjong-cascade-v26-effv2s-v4-20261027/`。

`model.json`保存`architecture=yolo-efficientnet-v2-s-cascade`、`format=two-stage-pt`和完整42类映射。它及权重已被winner lock绑定，部署时不能修改；部署地址、public-source链接等写入外部release receipt。

- `detector.pt`：20,342,469字节，SHA256 `92f4091baa4dea3af611ee8b1e9211b49a58d72fdd8732cddff0e2344cc12afb`。
- `classifier.pt`：81,817,567字节，SHA256 `bc1ce8fa421ae3a81636f2de31da91b376a7c6c13160fdc79afff4bfa5820cfc`。
- classifier从244,060,605字节的原始checkpoint中仅移除optimizer/scheduler；导出后每个模型tensor与源checkpoint逐一`torch.equal`验证，未量化或转换权重精度。
- classifier源训练90分钟超时，保留`timed_out_checkpoint_recovery`证明；没有伪造`finishedAt`，没有把原训练标成完成。
- 本次仅推理/评估/打包，没有执行train、backward或optimizer step，也没有更改v5冻结源、队列或原始数据。

推理依赖为Ultralytics8.4.189、torch2.8、torchvision0.23。Ultralytics相关AGPL要求由发布流程处理；模型评估本身不改变授权。Windows4070Ti验证不能取代Modal Linux GPU的实际接口联调。

权重metadata及raw pickle已只读扫描，未检出私有home、已知用户名/机名、LAN IP、私有源仓库或常见密钥标记；detector的8个绝对路径均在`D:/mahjong/`，classifier无绝对路径。记录见[weight-privacy-audit-20261027.json](./weight-privacy-audit-20261027.json)。不将含逐图记录的完整validation报告直接作为public metrics发布。

完整选择摘要与哈希：[release-20261027.json](./release-20261027.json)。包内包含GPU验证报告、sealed test报告与winner lock。`productionValidated`继续为false。

## 可复现的后评估

使用独立代码快照`D:/mahjong/code-release-eval-r3-20261027`（bundle SHA256 `9eabb2485be48c652019dc1b225e091225926693cea88b8e474467c2bc8f15f2`）。原v5训练目录不修改。

- `ml.eval.recover_terminal`：完整/有终态证明的YOLO与cascade权重验证；CPU模式。
- `ml.eval.recover_rfdetr`：终态RF权重的GPU验证。
- `ml.eval.release_candidate`：已选择配对的GPU全valid复核和CPU/GPU统计对照。
- `ml.train.freeze_yolo --strip-classifier-training-state`：与报告绑定、无损去训练状态打包。
- `ml.eval.sealed_test lock`：只根据GPU validation锁定winner；不读取test标签。
- `ml.eval.sealed_test test`：仅一个固定工作点、全test、独占打开ledger，拒绝阈值网格与重开。

发现并修复了COCO AP评估中的annotation ID0与“unmatched=0”哨兵冲突。新`ml.eval.coco_metrics`只在评估器局部复制为1..N，不改变原数据；Windows真实faster-coco-eval完美框回归AP=1.0且原id0保持不变。主整手指标不依赖COCO AP，这项修复未用于改变winner。
