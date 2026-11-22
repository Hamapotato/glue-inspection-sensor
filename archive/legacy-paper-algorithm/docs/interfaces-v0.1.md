# 接口契约

> 目的：让传统链路（C 组）与深度学习（B 组）可以并行开发，互不阻塞。
> 实现可以替换，接口签名与语义不能随意改。

---

## 1. 全局约定

| 项 | 约定 |
|---|---|
| 图像 | `np.ndarray`，单帧；灰度 `(H,W) uint8` 或彩色 `(H,W,3) uint8`（BGR） |
| 坐标 | 像素坐标系，原点左上，`x` 向右，`y` 向下 |
| 框 | `(x1, y1, x2, y2)`，且 `x1 ≤ x2`、`y1 ≤ y2`，浮点，允许落在图像边界上 |
| 分数 | `[0,1]`，越大越可信；IoU 家族的相似度可为负（GIoU/DIoU/CIoU） |
| 单位 | 长度内部一律用像素；转毫米只在 `measure` 输出时做 |
| 时间 | 毫秒，浮点 |

---

## 2. 数据结构

定义在 `algo/types.py`：

```python
Box(x1, y1, x2, y2, score=1.0, label=0)
MeasureResult(area_px, length_px, width_px, width_mm, n_components, bbox,
              min/max/width_std, curve_deviation, break_windows,
              scan_windows, centerline_segments)
ReflectionResult(has_reflection, boxes, max_score)
Verdict(code, severity, message, details)
FrameResult(verdict, measure, reflection, roi, timings_ms)
```

`Verdict.code` 枚举：`OK` / `BROKEN` / `TOO_NARROW` / `TOO_WIDE` /
`WIDTH_UNSTABLE` / `BENT` / `WARN_NARROW` / `WARN_WIDE` / `REFLECTION_SUSPECT`。
`Verdict.severity`：`none` / `warn` / `alarm`。

---

## 3. 三个核心接口

### 3.1 RoI 定位

```python
class RoILocator(Protocol):
    def predict(self, image: np.ndarray) -> list[Box]:
        """返回按 score 降序的候选框；无目标时返回空列表。"""
```

- 实现：`RPNLocator`（论文 3.1）/ `HeuristicLocator`（占位，阈值+轮廓）
- 约束：至少返回 0 个框；多于 1 个框时上层按论文判定逻辑判"断胶"（见 decide）
- 缺模型文件时：抛 `ModelNotAvailableError`，由 pipeline 决定降级

### 3.2 反光检测

```python
class ReflectionDetector(Protocol):
    def predict(self, image: np.ndarray) -> ReflectionResult:
        """has_reflection=True 当且仅当存在 score ≥ threshold 的框。"""
```

- 实现：`YOLO5FLDetector`（论文 3.3）/ `NullReflector`（永远 False）/ `HeuristicReflector`（高亮区域）

### 3.3 量测

```python
class Measurer(Protocol):
    def measure(self, roi_img: np.ndarray, seeds: np.ndarray | None = None) -> MeasureResult:
        """roi_img 为裁出的胶条图像；seeds 为置信点 (N,2) 的 (x,y)。"""
```

- 内部顺序固定：去噪 → 分割 → 形态学 → BFS 面积 + 骨架长度 → 宽度
- `width_mm` 由 `Calibration.scale` 换算；未标定时 `width_mm = 0` 且 `details` 标记 `uncalibrated`

---

## 4. 标定接口

```python
@dataclass
class Calibration:
    K: np.ndarray | None          # 3x3 内参
    dist: np.ndarray | None       # 畸变
    rms: float                    # 重投影误差 RMS
    mm_per_px: float | None       # 工作距离处尺度
    H: np.ndarray | None          # 平面单应（3x3），可选

    def px_to_mm(self, px: float) -> float: ...
```

- `calibrate_from_images(images, pattern, square_mm)`：棋盘格标定
- `calibrate_from_points(obj_pts, img_pts, image_size)`：已知点对（测试/离线用）
- 未标定时 `mm_per_px=None`，上层必须能降级运行

---

## 5. 配置键（与 `configs/default.yaml` 一一对应）

```
input:      {resize_long_side, to_gray}
calib:      {enabled, mm_per_px, pattern:[cols,rows], square_mm}
preprocess: {denoise:{method,wavelet,level,mode}, segment:{method}, morph:{ksize,order}}
measure:    {bfs_connectivity, min_area_px, skeleton:{method},
             width:{standard_mm, warn_delta_mm, alarm_delta_mm},
             scan:{enabled,step_px,window_length_px,max_ray_px,ray_step_px,
                   break_min_missing_windows,width_range_alarm_mm,
                   width_std_alarm_mm,curve_warn_mm,curve_alarm_mm}}
locate:     {enabled, type, weights, score_thr, nms:{type,thr}, anchors:{k,iters}}
reflect:    {enabled, type, weights, score_thr, input_size, high_value, min_area_px}
decide:     {broken_min_rois, reflection_override}
runtime:    {device, half}
synthetic_dataset:
            {version,image_size,samples,seed,frames_per_object,centerline_points,
             split,scenario_weights,scene,glue,defects,camera}
visiontrain_export:
            {image_type,contour_epsilon_px,min_contour_area_px}
```

约定：**代码里不允许出现业务阈值字面量**，全部从配置读。

`synthetic_dataset` 仅控制 VisionTrain 合成训练集生成。所有输出均带
`source=synthetic` 与 `acceptance_eligible=false`，不得用于真实指标验收。
语义类别固定为 `background=0`、`glue=1`；场景固定为正常胶条、纯背景、
断胶、过窄、过宽、宽窄不一致和弯曲。`defects` 段只控制断胶生成参数，
不新增分割类别，也不生成气泡、拉丝、污染或反光样本。

`visiontrain_export` 仅控制掩膜到 VisionTrain COCO 风格多边形标注的转换；
转换成功不等于 VisionTrain GUI 回读和训练验收成功。导出类别固定为
`background` 和 `glue`。

---

## 6. 结果输出

`FrameResult.as_dict()` 即对外 JSON，字段稳定，可直接入库/上报：

```json
{
  "verdict": {"code":"TOO_NARROW","severity":"alarm","message":"...","details":{}},
  "measure": {"area_px":12345.0,"length_px":312.0,"width_px":39.6,"width_mm":7.42,
              "n_components":1,"bbox":{"x1":0,"y1":0,"x2":0,"y2":0,"score":1.0,"label":0},
              "min_width_mm":7.1,"max_width_mm":7.8,"width_std_mm":0.2,
              "curve_deviation_mm":1.3,"break_windows":0,"missing_edge_windows":0,
              "scan_windows":[{"index":0,"cx":120.0,"cy":80.0,"angle_deg":12.0,
                               "left":[100.0,75.0],"right":[140.0,85.0],
                               "midpoint":[120.0,80.0],"valid":true}],
              "centerline_segments":[[[100.0,75.0],[120.0,80.0]]]},
  "reflection": {"has_reflection":false,"boxes":[],"max_score":0.0},
  "roi": {"x1":0,"y1":0,"x2":0,"y2":0,"score":0.99,"label":0},
  "timings_ms": {"preprocess":3.1,"locate":12.0,"measure":4.2,"total":19.3}
}
```

---

## 7. 降级策略（必须实现，否则没法在缺模块时跑）

| 缺失 | 降级行为 | 影响 |
|---|---|---|
| 标定文件 | `width_mm=0`，`details.uncalibrated=True` | 只能出像素宽度 |
| RPN 权重 | 用 `HeuristicLocator` 裁整图 | 定位指标不可用 |
| 反光权重 | `NullReflector`（永不判反光） | 反光分支不可用 |
| GPU | CPU 推理 | 速度慢，功能不变 |

降级必须写进结果字段（`details`），禁止静默降级。
