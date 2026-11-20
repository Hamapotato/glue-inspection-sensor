# VisionTrain 涂胶分割数据准备

> 状态：合成数据生成器已实现
> 适用范围：VisionTrain 2.3 / VisionMaster 4.4 流程验证与预训练
> 关联任务：A-07

## 1. 数据边界

Python 生成的数据全部标记为 `source=synthetic`，并在 `dataset.json` 中写入
`acceptance_eligible=false`。这些数据可用于：

- 验证图像、掩膜和标签格式；
- 跑通 VisionTrain 训练和 VisionMaster 部署；
- 做真实数据到位前的模型预训练；
- 构造断胶、气泡、拉丝和污染等困难样本。

合成数据不得用于：

- 报真实召回率、误检率或宽度误差；
- 代替真实相机、镜头、光源和标定验证；
- 作为冻结测试集完成项目验收。

## 2. 生成命令

默认配置生成 60 张 1440 x 1080 灰度图：

```powershell
py -3 scripts/generate_visiontrain_dataset.py `
  --out outputs/synthetic/visiontrain-glue-v1 `
  --samples 60 `
  --seed 20261120
```

输出目录必须不存在或为空，生成器不会静默覆盖历史数据。参数集中在
`configs/default.yaml` 的 `synthetic_dataset` 段。

## 3. 输出结构

```text
outputs/synthetic/visiontrain-glue-v1/
├─ images/<image_id>.png
├─ masks/
│  ├─ glue/<image_id>.png
│  ├─ defects/<image_id>.png
│  └─ hard_background/<image_id>.png
├─ labels/<image_id>.json
├─ splits/{train,val,test}.txt
├─ manifest.csv
└─ dataset.json
```

掩膜约定：

| 文件 | 取值 |
|---|---|
| `glue` | `0` 背景，`255` 胶条 |
| `defects` | `0` background，`1` break，`2` bubble，`3` stringing，`4` contamination |
| `hard_background` | `0` 普通背景，`255` 反光等困难背景 |

所有图像和掩膜使用同一随机变换，不进行二次缩放。`manifest.csv` 记录
`object_id`、`batch`、`split`、随机种子和 `acceptance_eligible=0`。

## 4. 导入 VisionTrain

本机 VisionTrain 2.3 手册说明：

- 可通过文件夹导入图像；
- 若文件夹包含当前算法模块可识别的标注信息，会一并导入；
- VisionTrain 自己导出的数据集包含标注信息，可以重新加载；
- 手册未说明直接支持任意 COCO、VOC 或 PNG 掩膜目录。

因此正式批量转换前必须先做一个最小验证：

1. 在 VisionTrain 的 `DL图像分割` 中导入 20 张合成图；
2. 手工标注 1～2 张后导出数据集；
3. 检查其数据集包内的图片、JSON 和掩膜结构；
4. 根据实际结构增加 Python 转换器；
5. 只对 2～3 张做导入回读，确认标签和掩膜没有错位后再批量转换。

在没有完成上述回读前，不得声称 Python 生成的标签可以直接被 VisionTrain 自动识别。

## 5. 训练拆分

生成器按 `object_id` 分配 train/val/test。同一 `object_id` 的所有帧只进入一个集合。
VisionTrain 的“自动划分”不能替代这一分组规则。

测试集一经开始调参就必须冻结。合成测试集只用于冒烟，真实项目验收仍需另行建立
按工件或批次划分的真实冻结测试集。
