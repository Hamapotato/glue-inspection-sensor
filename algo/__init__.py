"""涂胶检测算法复现框架（对应学位论文第 2–4 章）。

模块划分
--------
- types      : 数据结构（框、检测结果、量测结果、判定结论）
- config     : YAML 配置加载
- geometry   : 框运算、IoU/GIoU/DIoU/CIoU、编解码、NMS/CIoU-NMS、K-means anchors
- preprocess : 小波去噪、中值滤波、Otsu/双峰/全局阈值分割、开闭运算
- measure    : BFS 面积、骨架细化、面积/长度求平均宽度
- calib      : 相机标定、重投影误差、像素→毫米映射
- models     : RoI 定位（RPN）与反光检测（YOLOv3-5FL+SPP）的接口与占位实现
- decide     : 断胶 / 过宽过窄 / 反光复判的判定规则（论文 5.3.3）
- pipeline   : 单帧端到端流水线
- metrics    : 评测指标（IoU、召回、MAE/标准差/最大误差）
- synth      : 合成样本生成（无真实数据时跑通链路与测试用）
- synthetic_dataset : VisionTrain 涂胶分割训练集生成（仅合成预训练/冒烟）
"""

__version__ = "0.1.0"
