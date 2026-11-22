# glue-inspection-sensor

基于机器视觉的车身涂胶在线检测仓库。

当前唯一算法主线：

```text
胶条/背景分割模型
  -> 概率图二值化与形态学后处理
  -> 沿涂胶方向生成旋转扫描矩形
  -> 每个矩形内定位左右胶条边缘点
  -> 左右边缘点中点形成实际中心曲线
  -> 断胶、宽窄不一致、弯曲判定
```

模型只负责胶条像素分割；缺陷由扫描窗中的边缘、宽度和中心线几何量测得到。

GitHub：[Hamapotato/glue-inspection-sensor](https://github.com/Hamapotato/glue-inspection-sensor)

## 当前状态

- [x] 确定“胶条分割模型 + 旋转扫描矩形”为唯一算法主线
- [x] 完成分割模型接口、二值后处理和主线流水线
- [x] 完成旋转扫描矩形、边缘中点曲线和缺陷判定
- [x] 扫描矩形横向宽度默认至少为实测胶宽的 3 倍
- [x] 归档旧 RPN、YOLO 反光、面积/骨架平均宽度等实现
- [x] 生成 `visiontrain-glue-binary-v5` 合成训练与导出数据
- [x] 完成 VisionMaster 首轮方案运行、正式命名和部署资产归档
- [ ] 训练并部署经真实数据验证的胶条/背景分割模型
- [ ] 完成真实相机标定、尺寸验证和缺陷验收

## 目录

```text
.
├── algo/                    # 分割接口、扫描量测、判定、标定
├── configs/                 # 主线参数
├── scripts/                 # 数据集生成、导出、标定和演示入口
├── tests/                   # 主线单元测试
├── deployment/              # VisionMaster 方案、模型和配套验证集（二进制本地保存）
├── docs/
│   ├── README.md            # 文档索引
│   └── algo/                # 当前技术路线文档
├── data/                    # 数据规范与 manifest 模板
├── archive/
│   ├── legacy-paper-algorithm/
│   └── non-mainline-docs/
└── references/              # 版权与资料说明
```

## 快速入口

- [文档索引](docs/README.md)
- [算法主线方案](docs/algo/plan.md)
- [接口契约](docs/algo/interfaces.md)
- [旋转扫描矩形量测](docs/algo/scanline-measurement.md)
- [单相机标定计划](docs/algo/calibration-plan.md)
- [VisionTrain 数据准备](docs/algo/visiontrain-data-preparation.md)
- [VisionTrain 训练 SOP](docs/algo/visiontrain-training-sop.md)
- [VisionMaster 部署 SOP](docs/algo/visionmaster-deployment-sop.md)
- [VisionMaster 纯 VM 主线](docs/algo/visionmaster-pure-vm-mainline.md)
- [VisionMaster 拖拽端到端 SOP](docs/algo/visionmaster-dragdrop-end-to-end.md)
- [M06 首轮部署包](deployment/visionmaster/glue_m06_inspection_v1/README.md)
- [M06 同轨迹验证数据集](docs/algo/m06-matched-validation-dataset.md)
- [任务清单](docs/algo/tasks.md)
- [旧论文算法归档](archive/legacy-paper-algorithm/README.md)
- [非主线文档归档](archive/non-mainline-docs/README.md)
- [AI agent 协作规范](AGENTS.md)

## 常用命令

```powershell
# 单元测试
py -3 -m unittest discover -s tests -v

# 合成掩膜主线演示
py -3 scripts/demo_scanline.py --out outputs/demo

# 生成 v5 合成数据
py -3 scripts/generate_visiontrain_dataset.py `
  --out outputs/synthetic/visiontrain-glue-binary-v5-500 `
  --samples 500 `
  --seed 20261121

# 导出 VisionTrain 二分类标注
py -3 scripts/export_visiontrain_annotations.py `
  --source outputs/synthetic/visiontrain-glue-binary-v5-500 `
  --out outputs/visiontrain-export/visiontrain-glue-binary-v5-500-full

# 生成 M06 同轨迹验证集（1 张模板 + 19 正常 + 80 异常）
py -3 scripts/generate_m06_validation_dataset.py `
  --out outputs/m06-validation/glue-m06-matched-v1-100
```

## 验证边界

当前已有 VisionMaster 4.4 首轮方案及 VM 440 CPU 分割模型，并已在用户人工
运行中观察到断胶、过宽、过窄和气泡响应。尚无逐类命中数、误报率、耗时等
冻结指标；M06 验证集仍是合成数据，真实相机胶宽和缺陷验收也尚未完成。
