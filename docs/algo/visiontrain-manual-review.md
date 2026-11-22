# VisionTrain 胶条/背景二分类人工校验

> 状态：`visiontrain-glue-binary-v5-500` 已生成，等待 VisionTrain GUI 回读与人工复核
> 关联任务：M-11
> 数据边界：`source=synthetic`、`acceptance_eligible=false`

本文区分三种“正确”：

1. **格式正确**：VisionTrain 能导入图片、标签和 JSON；
2. **标注正确**：`glue` 多边形与胶条一致，断口被排除，背景图没有 `glue`；
3. **业务有效**：模型在真实产线上达到指标。前两项通过也不等于第三项成立。

## 1. 数据语义

新版数据只有两个类别：

| ID | 类别 |
|---:|---|
| 0 | `background` |
| 1 | `glue` |

断胶、过窄、过宽、沿程宽度不一致和弯曲只作为生成场景及标签元数据，
不新增分割类别。

## 2. 校验目录

建议先导出全部样本：

```text
D:\VMProjects\visiontrain_glue_binary_v5_500_full
```

再从各场景中选 1～2 张组成 roundtrip 子集。具体 `image_id` 必须根据
`manifest.csv` 的 `scenario` 列填写，不能沿用旧版
`visiontrain-glue-v1` 的样例 ID。

## 3. 第一轮：验证格式

1. 启动 VisionTrain 2.3.0。
2. 新建一个干净方案。
3. 添加 `DL图像分割` 模块。
4. 确认标签列表只有 `background` 和 `glue` 两个业务类别。
5. 导入 roundtrip 目录最外层，不要只选择单张 PNG。
6. 检查图像数量、标签名称和同名 JSON 是否全部导入。
7. 完整数据应显示 500 张，train/val/test 为 350/75/75。

## 4. 第二轮：检查二值语义

每个场景的标注标准：

| `scenario` | 正确标准 |
|---|---|
| `glue` | 完整胶条为 `glue`，其余区域为背景 |
| `background` | 不应出现任何 `glue` 多边形 |
| `break` | 左右胶条为 `glue`，中间断口不能标成 `glue` |
| `narrow` | 窄胶仍应完整标注为 `glue`，宽度由扫描矩形量测 |
| `wide` | 宽胶仍应完整标注为 `glue`，宽度由扫描矩形量测 |
| `width_unstable` | 宽度变化区域都保留为 `glue`，供局部宽度统计 |
| `curve` | 弯曲胶条仍应完整标注为 `glue`，供中心线拟合 |

重点检查：

- 图像中不能出现灰色工件板；
- 背景只能包含渐变、纹理和噪声；
- 断口是否真实形成非胶空洞；
- 宽窄变化和弯曲是否仍然完整落在 `glue` 掩膜内；
- 标注是否与源 `masks/glue` 对齐；
- 缺陷类型是否已写入 `labels/*.json` 的 `defects` 字段。

## 5. 通过条件

- [ ] 二分类类别仅为 `background/glue`；
- [ ] 纯背景图能够导入并显示为已标注；
- [ ] 胶条图的多边形与源掩膜对齐；
- [ ] 断口未标成 `glue`；
- [ ] 宽窄变化和弯曲没有造成掩膜断裂；
- [ ] 原始图像中不存在工件板；
- [ ] 完整数据回读为 500 张，且分组保持 350/75/75；
- [ ] 未把合成数据写成真实指标。

人工复核通过后，才能开始 VisionTrain 训练。GUI 回读是否成功仍需单独记录。
