# 接口契约

## 1. 分割模型接口

```python
class GlueSegmenter(Protocol):
    def predict(self, image: np.ndarray) -> SegmentationOutput: ...
```

`SegmentationOutput` 字段：

| 字段 | 含义 |
|---|---|
| `mask` | 概率图或二值掩膜 |
| `source` | 模型或数据源标识 |
| `threshold` | 模型建议阈值，流水线可由配置覆盖 |
| `fallback` | 是否为开发 fallback |
| `score_map` | 可选原始概率图 |

流水线会把概率统一到 `[0,1]`，按 `segmentation.threshold` 二值化，再执行
`segmentation.morph` 配置的开闭运算。

无分割模型时不得静默改用旧算法。只有显式设置：

```yaml
segmentation:
  mode: development_threshold
```

才允许启用仅供冒烟的阈值 fallback，并在结果中记录降级。

## 2. 检测器入口

```python
inspector = GlueInspector(
    cfg=config,
    segmenter=model_segmenter,
    calib=calibration,
)
result = inspector.run(image)
```

`GlueInspector` 不再接收 RoI 定位器或反光检测器。

## 3. 扫描窗

`ScanWindow` 关键字段：

| 字段 | 含义 |
|---|---|
| `cx, cy` | 扫描窗中心，原图像素坐标 |
| `angle_deg` | 沿局部涂胶方向的旋转角 |
| `length_px` | 沿胶条方向的窗口长度 |
| `window_width_px` | 沿法向的窗口宽度，默认至少 3 倍胶宽 |
| `width_px` | 左右边缘点之间的实际胶宽 |
| `left/right` | 左右边缘点 |
| `midpoint` | 左右边缘点中点 |
| `valid` | 是否找到左右两个边缘点 |
| `break_detected` | 是否属于断胶窗 |

## 4. 输出结构

`FrameResult.as_dict()` 示例：

```json
{
  "verdict": {
    "code": "OK",
    "severity": "none",
    "message": "合格",
    "details": {}
  },
  "measure": {
    "width_mm": 10.2,
    "min_width_mm": 9.9,
    "max_width_mm": 10.5,
    "width_std_mm": 0.16,
    "curve_deviation_mm": 1.1,
    "break_windows": 0,
    "scan_windows": [
      {
        "window_width_px": 120.0,
        "left": [80.0, 100.0],
        "right": [120.0, 100.0],
        "midpoint": [100.0, 100.0]
      }
    ],
    "centerline_segments": []
  },
  "segmentation": {
    "source": "visiontrain_model",
    "threshold": 0.5,
    "fallback": false,
    "foreground_ratio": 0.08
  },
  "degradations": []
}
```

## 5. 配置键

```yaml
segmentation:
  mode: external
  model_path: null
  threshold: 0.5
  morph:
    ksize: 5
    order: open-close

measure:
  mask_connectivity: 8
  min_area_px: 200
  scan:
    enabled: true
    step_px: 12
    window_length_px: 24
    window_width_factor: 3.0
    max_ray_px: 180
    ray_step_px: 0.5
    break_min_missing_windows: 1
    centerline_smooth_window: 7
    width_range_alarm_mm: 1.2
    width_std_alarm_mm: 0.30
    curve_warn_mm: 2.0
    curve_alarm_mm: 3.0
```

### 5.1 M06 同轨迹验证集配置

```yaml
m06_validation:
  version: m06-glue-matched-v1
  trajectory_id: m06_fixed_path_v1
  seed: 20261121
  total_samples: 100
  image_size: [1080, 1440]
  pixel_format: Mono8
  counts:
    reference_good: 1
    normal: 19
    break: 20
    narrow: 20
    wide: 20
    bend: 20
  fixed_trajectory: {}
  variants: {}
```

`fixed_trajectory` 中所有样本共用同一条标准胶路；`variants` 控制断胶长度、
宽度缩放和弯曲幅值。完整参数见 `configs/default.yaml`。这些参数只用于
生成合成验证数据，不是产线业务阈值。

## 6. 降级规则

| 缺失 | 行为 |
|---|---|
| 分割模型 | 默认报错；仅显式配置开发 fallback 时运行并记录降级 |
| 标定 | 输出像素宽度，跳过毫米报警，记录 `uncalibrated` |
| GPU | 允许 CPU，但必须重新统计耗时 |
