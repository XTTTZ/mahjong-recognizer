# 补充数据候选审查

核验日期：2026-10-26。以下是本地研究暂存，**尚未混入训练/验证/封存test**，也没有改动Roboflow原项目。样本URL、字节数和SHA256见[supplemental-candidates.json](./supplemental-candidates.json)。

| 来源 | 已核验内容 | 决策 |
| --- | --- | --- |
| [ShinZ手机实拍](https://www.kaggle.com/datasets/shinz114514/mahjong-hand-photos-taken-with-mobile-camera) | Kaggle公共API显示MIT、v1、2025-07-04更新、2,229,140,518原始字节；完整分页文件清单为257份PNG，没有标注文件；抽取前三张查看 | 值得继续：真实自动麻将桌、手牌/副露/弃牌/牌墙并存；先暂存原图，需穷尽标注或人工指定ROI及其标注后才能用于监督训练 |
| [Roman Nguyen Mahjong](https://universe.roboflow.com/roman-nguyen/mahjong-vtacs) | Universe页面4378张、34类、MIT；公共图片接口抽看6张，含单牌、双牌及18/25张散牌 | 保留候选；样式和地板散牌有补充价值，但本次没有得到已核验的完整COCO框，不能只凭每类计数代替bbox真值 |
| [Camerash麻将单牌](https://github.com/Camerash/mahjong-dataset) | 仓库MIT；README说明主要来自Google/eBay/Alibaba，240×320单牌 | 不用它替代真实整手验证；与现有网络单牌来源还可能重合 |

前三份Kaggle样本尺寸1706×1279、4030×2574、4087×2883；抽样确实来自麻将桌实拍，但只有3张不能代表全部257张的质量。它们包含框外其他牌，不能把底部手牌自动当作唯一目标，也不能把未标区域当负例。

9份样本均未与现有11921张快照产生字节完全相同的SHA256。Kaggle三份的最近64位dHash距离分别17、16、17；Roman六份距离8–15。该检查只排查明显重复，**不能证明来源/拍摄会话独立**；旋转、裁切、压缩与同场景近邻仍需复核。任何后续标注加入都要按来源和会话分组，并优先避免污染现有test。

注意：对公开URL调用通用`versions_get`时，两个请求的返回ID实际落在`mahjong-anaun/...`，不是请求的原作者workspace；返回版本还有640方形拉伸与预增强。因此本次不把这些返回值当成外部独立数据证明，也不直接触发导出/合并。公共图片检索与网页许可用于候选审查，真正导入需要来源一致的完整快照。

本地样本与联系页位于`artifacts/data/supplement-review/`，受Git忽略。完整Kaggle原图包单独存放在`artifacts/data/supplemental/kaggle-mobile-v1/`，已完成下载：ZIP 2,227,358,825 bytes，SHA256 `35c076754dbac0a3d4345780cf731272f9df2b4e1168d0222b0b161f611044df`，全部257份PNG已通过ZIP CRC；同目录snapshot.json保留元数据；有照片不等于有可训练标注。
