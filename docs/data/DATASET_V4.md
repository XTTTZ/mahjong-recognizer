# V4：原图范围的 42 类牌面数据

状态：2026-10-26，主集和扩展集均已在 Windows 重建并校验。没有启动新一轮训练。

| 集合 | 主集图数 | 扩展集图数 | 主集框数 | 主集至少10牌图数 |
| --- | ---: | ---: | ---: | ---: |
| train | 2,629 | 4,671 | 17,868 | 1,114 |
| valid | 150 | 150 | 1,396 | 82 |
| test | 121 | 121 | 916 | 47 |

扩展集训练框31,545个、至少10牌图2,013张；两版验证和测试的标注、图片字节相同。42类在每个split均有实例，花牌仍明显较少。新实验从官方预训练重新开始，禁止续训旧v2/v3权重。

原始Roboflow v2 ZIP含11,921张、106,141框，按原样保存。v3过于保守地只纳入207张训练图，不能代表其余图全部不可用。V4重新审查原图标注范围、同原图增强和分组，保留原图像素，不再按真值框裁整手ROI。目标是图片内所有清晰可辨的正面牌；背面作为背景，不纳入42类。明显漏标正面、标注范围混杂、宽高不符和无法确认范围的场景暂不纳入主实验。

V4主集限制每个原图身份最多12个已有变体；扩展集放开这个上限，用于检验更多预增强的收益。同源增强绝不拆到不同split，验证/测试只取canonical原图。训练图数不是独立拍摄数：估计约372个原始照片身份，保守合并后263个训练来源组。不同来源组可能仍共享设备、牌桌和麻将样式，因而该验证集仍是研究性holdout，不能替代独立棋牌室手机实拍测试。

本地V2审计所生成的research test来源家族整组保护，即使其他增强变体原来在train也不放回训练。这里的V2指本地审计/分组版本，并非Roboflow版本2的原始train/valid/test目录。公开拼合数据的原始目录名不决定本地实验归属；以冻结manifest的`split`、`group_id`和图片SHA为准。没有为了数量把本地V2研究test家族移入训练。12张图的标注仅修复不超过0.02像素的边界舍入，修复前后均记录；130张图像/标注宽高不符者仍排除。代表图contact review不等于人工逐框重标全部字形，剩余错标风险需由错误分析和反馈逐步消除。

冻结来源与逐图去向见 `final-dataset-v4.json`、`final-dataset-v4-extended.json`、`split-v4*.jsonl`、`review-v4.json`。主集manifest SHA256为 `fb6f36f3db767e8ad467e5c82e7a732143e046a0c60c25dd195822e6d85c4e03`；数据准备会复验源ZIP、manifest、图片及COCO hash，拒绝静默替换。

Windows位置：

- `D:/mahjong/data/prepared-v4-main`
- `D:/mahjong/data/prepared-v4-extended`
- `D:/mahjong/data/smoke-v4`（96 train/24 valid，仅历史小批验证使用）

可复现重建（输出目录必须是新目录）：

```powershell
python -m ml.data.materialize_v4 --archive D:/mahjong/data/roboflow-v2-coco.zip --manifest docs/data/split-v4.jsonl --metadata docs/data/final-dataset-v4.json --out D:/mahjong/data/prepared-v4-main-new --source-root D:/mahjong/data/roboflow-v2-original
```

Kaggle手机实拍257张、MIT来源ZIP已冻结，但没有人工标注，当前没有混入任何split。后续应优先把真实手机失败样本规范标注并按拍摄会话整体留出，避免只增加同一照片的增强数量。
