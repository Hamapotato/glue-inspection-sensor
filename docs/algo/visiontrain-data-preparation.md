# VisionTrain 胶条/背景二分类数据准备

> 状态：v5 五百张数据与完整导出已生成；VisionTrain GUI 回读和人工复核待完成
> 适用范围：VisionTrain 2.3 / VisionMaster 4.4 流程验证与预训练
> 关联任务：M-11、M-12

## 1. 数据边界

新版合成数据只定义两个语义类别：

| ID | 类别 | 含义 |
|---:|---|---|
| 0 | `background` | 背景和断口等非胶区域 |
| 1 | `glue` | 可见胶条 |

画面不绘制工件板，胶条直接位于背景纹理上。生成器会构造断胶、过窄、过宽、
沿程宽度不一致和弯曲场景，但不会新增缺陷分割类别。缺陷类型只写入
`labels/*.json` 和 `manifest.csv` 元数据，便于筛选、复核和后续扩展。

Python 生成的数据全部标记为 `source=synthetic`，并在 `dataset.json` 中写入
`acceptance_eligible=false`。这些数据可用于：

- 验证图像、掩膜和标签格式；
- 跑通 VisionTrain 训练和 VisionMaster 部署；
- 做真实数据到位前的胶条/背景分割预训练；
- 用断胶、宽窄变化和弯曲场景检查模型稳定性。

合成数据不得用于：

- 报真实召回率、误检率或宽度误差；
- 代替真实相机、镜头、光源和标定验证；
- 作为冻结测试集完成项目验收。

## 2. 生成命令

本轮正式流程验证数据集生成 500 张 1440 x 1080 灰度图：

```powershell
py -3 scripts/generate_visiontrain_dataset.py `
  --out outputs/synthetic/visiontrain-glue-binary-v5-500 `
  --samples 500 `
  --seed 20261121
```

输出目录必须不存在或为空，生成器不会静默覆盖历史数据。参数集中在
`configs/default.yaml` 的 `synthetic_dataset` 段。

默认场景及权重：

| `scenario` | 画面内容 | 胶条掩膜 |
|---|---|---|
| `glue` | 背景上有一条完整胶条 | 胶条区域为 255 |
| `background` | 只有背景 | 全 0 |
| `break` | 胶条中有一段真实断口 | 两段胶条为 255，断口为 0 |
| `narrow` | 整条胶宽低于标准值 | 胶条区域为 255 |
| `wide` | 整条胶宽高于标准值 | 胶条区域为 255 |
| `width_unstable` | 沿涂胶方向宽度明显波动 | 胶条区域为 255 |
| `curve` | 中心线明显弯曲 | 胶条区域为 255 |

## 3. 输出结构

```text
outputs/synthetic/visiontrain-glue-binary-v5-500/
├─ images/<image_id>.png
├─ masks/glue/<image_id>.png
├─ labels/<image_id>.json
├─ splits/{train,val,test}.txt
├─ manifest.csv
└─ dataset.json
```

掩膜约定：

| 文件 | 取值 |
|---|---|
| `masks/glue` | `0` background，`255` glue |

标签中的 `classes` 固定为：

```json
{
  "background": 0,
  "glue": 1
}
```

`labels/*.json` 的 `defects` 字段记录生成缺陷的类型、像素面积、外接框和
相关几何参数。该字段是元数据，不是第三个语义类别。

所有图像和掩膜使用同一次随机生成，不进行二次缩放。`manifest.csv` 记录
`scenario`、`has_glue`、`has_defect`、`defect_types`、`split`、随机种子和
`acceptance_eligible=0`。

当前数据统计：

| 项目 | 数量 |
|---|---:|
| 总图像 | 500 |
| 含胶条图像 | 423 |
| 纯背景图像 | 77 |
| 胶条多边形 | 490 |
| train / val / test | 350 / 75 / 75 |

场景分布：正常 `193`、背景 `77`、断胶 `67`、过窄 `58`、过宽 `58`、
宽窄不一致 `18`、弯曲 `29`。

## 4. 导入 VisionTrain

VisionTrain 2.3 实际导出的单图标注格式为 COCO 风格。新版转换器使用：

- `background=1`；
- `glue=2`；
- 每张图旁的 `<image_id>.json` 保存 `glue` 多边形；
- 没有胶条的图使用空数组 `[]`，并在 `dataset.json` 中标记 `labeled=true`。

生成两张 GUI 回读样例：

```powershell
py -3 scripts/export_visiontrain_annotations.py `
  --source outputs/synthetic/visiontrain-glue-binary-v5-500 `
  --out D:/VMProjects/visiontrain_glue_binary_v5_500_roundtrip `
  --image-id syn_20261121_000000 `
  --image-id syn_20261121_000001
```

生成全部 500 张：

```powershell
py -3 scripts/export_visiontrain_annotations.py `
  --source outputs/synthetic/visiontrain-glue-binary-v5-500 `
  --out D:/VMProjects/visiontrain_glue_binary_v5_500_full
```

仓库内已经生成并可核验的导出目录：

```text
D:\WORK\glue-inspection-sensor\outputs\visiontrain-export\visiontrain-glue-binary-v5-500-full
```

GUI 回读步骤：

1. 在 VisionTrain 的 `DL图像分割` 中导入 roundtrip 目录；
2. 检查胶条图只出现 `glue` 多边形；
3. 检查纯背景图为已标注但不出现 `glue` 多边形；
4. 检查断胶场景中非胶区域没有被错误标成 `glue`；
5. 检查导出图像总数为 500，训练/验证/测试为 350/75/75；
6. 以上通过后再开始训练。

在 GUI 回读完成前，不得声称该目录已经在 VisionTrain 中完整验收。

## 5. 训练拆分

生成器按 `object_id` 分配 train/val/test。同一 `object_id` 的所有帧只进入一个集合。
VisionTrain 的“自动划分”不能替代这一分组规则。

测试集一经开始调参就必须冻结。合成测试集只用于冒烟，真实项目验收仍需另行建立
按工件或批次划分的真实冻结测试集。
