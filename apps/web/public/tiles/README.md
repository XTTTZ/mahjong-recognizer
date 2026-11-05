# Mahjong tile face assets

The 42 face illustrations are vendored from the **Hong Kong** set in
[samoheen/mahjong-tiles](https://github.com/samoheen/mahjong-tiles), frozen at commit
[`8b5a3cd35ef22bcdb1f7f09dc9926f733885f7b4`](https://github.com/samoheen/mahjong-tiles/tree/8b5a3cd35ef22bcdb1f7f09dc9926f733885f7b4/hongkong/svg).
The upstream [README](UPSTREAM_README.md) and [license](LICENSE.md) declare the
files **Public Domain** and link to [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
No account, external runtime, third-party font or image CDN is required.

These are transparent **face-only SVGs**, without a white tile body. Keep the
application's tile frame/background. Original dimensions are approximately
300×420 (green dragon296×420 and one other face301×420). A `viewBox` matching
each original width/height has been added for resolution-independent scaling;
paths, colors, clip paths and all other artwork remain unchanged. The brush
characters are outlines, so appearance is independent of installed fonts.

Local names deliberately follow this app's ontology rather than upstream order:

| Local class | Face | Upstream filename |
| --- | --- | --- |
| `1m`–`9m` | 一萬–九萬 | `08-characters-1.svg`–`16-characters-9.svg` |
| `1p`–`9p` | 一筒–九筒 | `17-circles-1.svg`–`25-circles-9.svg` |
| `1s`–`9s` | 一条–九条 | `26-bamboos-1.svg`–`34-bamboos-9.svg` |
| `1z` | 东 | `04-east-wind.svg` |
| `2z` | 南 | `05-south-wind.svg` |
| `3z` | 西 | `06-west-wind.svg` |
| `4z` | 北 | `07-north-wind.svg` |
| `5z` | 白板 | `01-white-dragon.svg` |
| `6z` | 发 | `02-green-dragon.svg` |
| `7z` | 中 | `03-red-dragon.svg` |
| `1h` | 春 | `35-spring.svg` |
| `2h` | 夏 | `36-summer.svg` |
| `3h` | 秋 | `37-autumn.svg` |
| `4h` | 冬 | `38-winter.svg` |
| `5h` | 梅 | `39-plum.svg` |
| `6h` | 兰 | `40-orchid.svg` |
| `7h` | 菊 | `41-chrysanthemum.svg` |
| `8h` | 竹 | `42-bamboo.svg` |

`back.svg` is a new, text-free green patterned tile back drawn for this project,
under the repository's AGPL-3.0-only license; it is not an upstream CC0 asset.
Hong Kong has no upstream back illustration. Every file's source path, original
and local SHA-256, dimensions and bytes are recorded in [manifest.json](manifest.json).

All42 upstream SVGs were parsed and checked against a drawing-element allowlist;
no scripts, event handlers, external image/font links or remote URL references
are included. The whole43-SVG set is226,774 bytes before transfer compression.
A contact-sheet review confirmed number-tile counts, the bird on one-bamboo,
real bamboo stalks on other bamboo tiles, seven honor faces and eight flowers.
