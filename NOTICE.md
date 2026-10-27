# Data and model attribution

Mahjong training images and annotations: [Mahjong dataset, mahjong-anaun / mahjong-lfhxf](https://universe.roboflow.com/mahjong-anaun/mahjong-lfhxf), version 2, published via Roboflow Universe under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated by the source and downloaded README. Source files are preserved. This project adds audit metadata, excludes invalid dimensions and exact duplicates, and creates local grouped research splits. Detailed modification records and example mappings are under `docs/data/`; images are not redistributed in this repository.

RF-DETR core code and pretrained Small/Medium/Large models: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr), Apache-2.0. Only the core Apache route is configured here; RF-DETR Plus model variants are not included. The exact pretrained asset URL and integrity checks are recorded in `infra/windows/verified/assets.json`.

RT-DETRv2 comparison route, if trained: [official implementation](https://github.com/lyuwenyu/RT-DETR) and [PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd), Apache-2.0. Its fixed revision and mapping are recorded by the training script.
