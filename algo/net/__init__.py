"""PyTorch 模型层（B1 RPN / B2 YOLOv3-5FL+SPP）。

模块
----
- ``common`` : 基础卷积块、SPP、anchor 生成、NMS（不依赖 torchvision）
- ``rpn``    : B1，区域生成网络（论文 3.1）
- ``yolo``   : B2，YOLOv3-5FL + SPP（论文 3.3）

设计约束（AGENTS.md）：
- torch 为**可选依赖**：本包只在调用方显式导入时加载；传统链路（P1）不依赖它。
- 本包不读取配置、不写文件，纯模型与前向计算；训练/推理编排在 ``algo/train.py``
  与 ``scripts/`` 中。
"""
