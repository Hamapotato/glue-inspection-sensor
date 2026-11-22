"""旋转扫描矩形胶条检测主线。

模块划分
--------
- types       : 扫描窗、量测结果和帧结果
- config      : YAML 配置加载
- segmentation: 胶条分割模型接口、二值化与形态学后处理
- maskops     : 连通域统计与骨架细化
- scanline    : 旋转扫描矩形、左右边缘点、中心线与局部宽度
- calib       : 相机标定、重投影误差、像素到毫米映射
- decide      : 断胶、宽窄不一致与弯曲判定
- pipeline    : 分割模型到扫描量测的单帧主线
- visualize   : 分割和扫描结果可视化
- synthetic_dataset : VisionTrain 胶条/背景合成训练数据
- m06_validation_dataset: M06 固定轨迹配对验证数据
- visiontrain_export: VisionTrain 标注导出

旧论文并列算法位于 ``archive/legacy-paper-algorithm/``，不参与主线。
"""

__version__ = "0.2.0"
