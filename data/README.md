# 数据目录规范

> 本目录存放**本地数据**。图像、标注、切分文件默认不入库（见仓库根 `.gitignore`），
> 只有 `manifest.csv` 与 `splits/` 允许提交，用于复现数据来源与划分。
>
> 标注与划分规则见 [docs/algo/p0-spec.md](../docs/algo/p0-spec.md) §2–§3。

## 目录结构

```
data/
├─ raw/<date>/<camera_id>/<image_id>.png     # 原图，禁止裁剪/压缩
├─ labels/
│  ├─ roi/<image_id>.json                    # {"boxes": [[x1,y1,x2,y2], ...]}
│  ├─ reflection/<image_id>.json             # 反光框，同格式
│  ├─ mask/<image_id>.png                    # 胶条掩膜（仅评估用）
│  └─ width.csv                              # 人工卡尺宽度真值
├─ splits/{train,val,test}.txt               # 每行一个 image_id
└─ manifest.csv                              # 采集参数台账（可提交）
```

## 命名

- `image_id`：`<日期>_<相机号>_<工件号>_<帧号>`，例如 `20261110_c1_obj0012_f0007`。
- 文件名与 `image_id` 必须一致，脚本按文件名关联标注。

## manifest.csv 字段

| 字段 | 说明 | 示例 |
| --- | --- | --- |
| `image_id` | 唯一标识，与文件名一致 | 20261110_c1_obj0012_f0007 |
| `path` | 相对 `data/raw` 的路径 | 20261110/c1/20261110_c1_obj0012_f0007.png |
| `camera_id` | 相机编号（三相机方案为 1/2/3） | 1 |
| `object_id` | 工件/胶段编号 | obj0012 |
| `batch` | 批次，用于按批划分 | B03 |
| `timestamp` | 采集时间 | 2026-11-10T14:03:22 |
| `exposure_us` | 曝光（微秒） | 5000 |
| `gain_db` | 增益（dB） | 0 |
| `pixel_format` | 像素格式 | Mono8 |
| `working_distance_mm` | 工作距离 | 120 |
| `light_mode` | 光源状态 | red-strobe |
| `glue_color` | 胶色 | black / white |
| `label` | 类别 | OK / BROKEN / TOO_NARROW / TOO_WIDE / REFLECTION |
| `ignore` | 1 表示该图标注存疑，评测跳过 | 0 |

模板见 [`manifest.template.csv`](manifest.template.csv)。

## width.csv 字段

| 字段 | 说明 |
| --- | --- |
| `image` | 图像文件名（与 `image_id` 一致） |
| `gt_width_mm` | 人工卡尺测量宽度（建议同一位置测 3 次取均值） |
| `object_id` | 工件编号 |
| `operator` | 测量人 |
| `tool` | 量具型号与分度值 |

## 划分规则（必须遵守）

1. **按 `object_id` / `batch` 划分**，不按帧随机划分，避免同一条胶条泄漏到训练与测试；
2. 反光样本**整组划分**（同一光照/工位条件归入同一侧）；
3. 划分结果写入 `splits/*.txt`，一次确定后不得因结果不好而更换（p0-spec §2、AGENTS.md §6.1）。
