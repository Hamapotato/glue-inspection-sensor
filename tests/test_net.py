"""B1/B2 模型测试（需要 torch；缺失时整组跳过）。

覆盖：前向形状、编解码、损失、候选框生成、NMS、权重存取、数据变换、
与 ``algo.models`` 的接口集成，以及一个"小样本过拟合"冒烟（验证可训练）。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    import numpy as np

    from algo.data import SyntheticGlueDataset, letterbox, scale_boxes
    from algo.net.common import ciou_nms_torch, giou_diou_ciou, make_anchors, nms_torch
    from algo.net.rpn import (RPN, RPNConfig, ProposalConfig, anchors_from_kmeans,
                              decode_boxes, encode_boxes, propose, rpn_loss)
    from algo.net.yolo import (YOLO5FL, YOLOConfig, decode_yolo, detect, yolo_loss,
                               yolo_targets)


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestRPN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = RPN()
        cls.model.eval()

    def test_forward_shapes(self):
        x = torch.zeros(1, 1, 160, 160)
        cls_logits, reg, anchors, feat = self.model(x)
        self.assertEqual(feat.shape[1], 256)
        hw = (160 // 16) ** 2
        self.assertEqual(cls_logits.shape, (1, hw * 9))
        self.assertEqual(reg.shape, (1, hw * 9, 4))
        self.assertEqual(anchors.shape, (hw * 9, 4))

    def test_encode_decode_roundtrip(self):
        anchors = torch.tensor([[10.0, 10, 50, 50], [20, 20, 80, 60]])
        targets = torch.tensor([[12.0, 14, 55, 58], [18, 25, 90, 70]])
        deltas = encode_boxes(anchors, targets)
        decoded = decode_boxes(anchors, deltas)
        self.assertTrue(torch.allclose(decoded, targets, atol=1e-5))

    def test_loss_finite(self):
        x = torch.zeros(1, 1, 160, 160)
        cls_logits, reg, anchors, _ = self.model(x)
        gt = torch.tensor([[20.0, 30, 120, 90]])
        losses = rpn_loss(cls_logits[0], reg[0], anchors, gt, mode="diou")
        self.assertTrue(torch.isfinite(losses["loss"]).item())
        self.assertGreater(float(losses["loss"].detach()), 0.0)

    def test_loss_empty_gt(self):
        x = torch.zeros(1, 1, 160, 160)
        cls_logits, reg, anchors, _ = self.model(x)
        losses = rpn_loss(cls_logits[0], reg[0], anchors, torch.zeros((0, 4)), mode="diou")
        self.assertTrue(torch.isfinite(losses["loss"]).item())
        self.assertEqual(float(losses["loss_reg"]), 0.0)

    def test_propose_returns_boxes(self):
        x = torch.zeros(1, 1, 160, 160)
        cls_logits, reg, anchors, _ = self.model(x)
        boxes, scores = propose(cls_logits[0], reg[0], anchors, (160, 160),
                                ProposalConfig(nms_type="ciou_nms", nms_thr=0.5))
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(boxes.shape[0], scores.shape[0])
        if boxes.numel():
            self.assertGreaterEqual(float(boxes[:, 0].min()), 0.0)
            self.assertLessEqual(float(boxes[:, 2].max()), 160.0)

    def test_propose_min_size_and_score(self):
        anchors = torch.tensor([[0.0, 0, 80, 80], [100.0, 100, 104, 104]])
        cls_logits = torch.tensor([5.0, 5.0])
        deltas = torch.zeros(2, 4)
        boxes, scores = propose(cls_logits, deltas, anchors, (160, 160),
                                ProposalConfig(min_size=8.0, nms_type="nms"))
        self.assertEqual(boxes.shape[0], 1)  # 4×4 的框被 min_size 过滤

    def test_anchor_count(self):
        anchors = make_anchors(4, 4, 16, (8, 16, 32), (0.5, 1.0, 2.0))
        self.assertEqual(anchors.shape, (4 * 4 * 9, 4))

    def test_explicit_anchors(self):
        anchors = make_anchors(4, 4, 16, anchors=torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
        self.assertEqual(anchors.shape, (4 * 4 * 2, 4))

    def test_kmeans_anchors(self):
        boxes = torch.tensor([[0, 0, 20, 10], [0, 0, 22, 12], [0, 0, 100, 40], [0, 0, 104, 42]])
        anchors = anchors_from_kmeans(boxes, k=2, seed=0)
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(w > 0 and h > 0 for w, h in anchors))

    def test_save_load_roundtrip(self):
        model = RPN()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rpn.pt"
            model.save(path)
            loaded = RPN.load(path)
        x = torch.zeros(1, 1, 160, 160)
        with torch.no_grad():
            a = model(x)[0]
            b = loaded(x)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestRPNTraining(unittest.TestCase):
    def test_loss_decreases_on_two_samples(self):
        torch.manual_seed(0)
        model = RPN(RPNConfig(width=8))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        x = torch.zeros(1, 1, 128, 128)
        gt = torch.tensor([[20.0, 40, 108, 88]])
        losses_seen = []
        for i in range(24):
            # 固定负样本采样，保证每步损失可比
            torch.manual_seed(1234)
            cls_logits, reg, anchors, _ = model(x)
            losses = rpn_loss(cls_logits[0], reg[0], anchors, gt, mode="diou")
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            losses_seen.append(float(losses["loss"].detach()))
        first = sum(losses_seen[:8]) / 8
        last = sum(losses_seen[-8:]) / 8
        self.assertLess(last, first, f"loss 未下降：first={first:.4f} last={last:.4f}")


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestYOLO(unittest.TestCase):
    def test_forward_5_levels(self):
        model = YOLO5FL(YOLOConfig(num_levels=5, use_spp=True)).eval()
        with torch.no_grad():
            outs = model(torch.zeros(1, 1, 320, 320))
        self.assertEqual(len(outs), 5)
        self.assertEqual([o.shape[2] for o in outs], [100, 400, 1600, 6400, 25600])
        self.assertEqual(outs[0].shape[-1], 6)  # 5 + 1 类

    def test_forward_3_levels(self):
        model = YOLO5FL(YOLOConfig(num_levels=3, use_spp=False)).eval()
        with torch.no_grad():
            outs = model(torch.zeros(1, 1, 320, 320))
        self.assertEqual(len(outs), 3)
        self.assertEqual([o.shape[2] for o in outs], [100, 400, 1600])

    def test_loss_finite_and_targets(self):
        model = YOLO5FL(YOLOConfig(num_levels=3, use_spp=True))
        x = torch.zeros(1, 1, 320, 320)
        outs = model(x)
        gt = torch.tensor([[60.0, 120, 260, 160]])
        targets = yolo_targets(outs, gt, (320, 320), anchors=model.anchors)
        losses = yolo_loss(outs, targets, (320, 320), anchors=model.anchors)
        self.assertTrue(torch.isfinite(losses["loss"]).item())
        self.assertGreater(float(losses["loss"].detach()), 0.0)
        # 每个 GT 只分配到一个正样本（最优层 + 最优 anchor）
        self.assertEqual(sum(int(t["mask"].sum()) for t in targets), 1)

    def test_decode_returns_boxes(self):
        model = YOLO5FL(YOLOConfig(num_levels=3)).eval()
        with torch.no_grad():
            outs = model(torch.zeros(1, 1, 320, 320))
            decoded = decode_yolo(outs, (320, 320), anchors=model.anchors, conf_thr=0.0)
        boxes, scores = decoded[0]
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(boxes.shape[0], scores.shape[0])

    def test_detect_nms_reduces(self):
        model = YOLO5FL(YOLOConfig(num_levels=3, conf_thr=0.0, nms_thr=0.3)).eval()
        out = detect(model, torch.zeros(1, 1, 320, 320), conf_thr=0.0)
        boxes = out[0]["boxes"]
        self.assertLessEqual(boxes.shape[0], model.cfg.max_det)

    def test_save_load_roundtrip(self):
        model = YOLO5FL(YOLOConfig(num_levels=3, use_spp=True)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yolo.pt"
            model.save(path)
            loaded = YOLO5FL.load(path)
        x = torch.zeros(1, 1, 320, 320)
        with torch.no_grad():
            a = model(x)
            b = loaded(x)
        for u, v in zip(a, b):
            self.assertTrue(torch.allclose(u, v, atol=1e-6))


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestNMS(unittest.TestCase):
    def test_nms_and_ciou_nms(self):
        boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(sorted(nms_torch(boxes, scores, 0.3).tolist()), [0, 2])
        self.assertEqual(sorted(ciou_nms_torch(boxes, scores, 0.3).tolist()), [0, 2])

    def test_similarity_modes(self):
        a = torch.tensor([[0.0, 0, 10, 10]])
        b = torch.tensor([[5.0, 5, 15, 15]])
        iou = giou_diou_ciou(a, b, mode="iou")
        self.assertAlmostEqual(float(iou[0]), 1 / 7, places=5)
        for mode in ("giou", "diou", "ciou"):
            self.assertTrue(torch.isfinite(giou_diou_ciou(a, b, mode=mode)).all().item())


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestData(unittest.TestCase):
    def test_letterbox_and_scale(self):
        img = np.zeros((320, 480), dtype=np.uint8)
        canvas, scale, pad = letterbox(img, (320, 320))
        self.assertEqual(canvas.shape, (320, 320))
        self.assertAlmostEqual(scale, 320 / 480, places=5)
        boxes = np.array([[60.0, 100, 420, 200]], dtype=np.float32)
        mapped = scale_boxes(boxes, scale, pad)
        self.assertAlmostEqual(float(mapped[0, 0]), 60 * scale + pad[0], places=4)

    def test_split_by_object_no_leak(self):
        from algo.data import split_by_object

        records = [{"image_id": f"img{i}", "object_id": f"obj{i // 2}"} for i in range(20)]
        splits = split_by_object(records, ratios=(0.6, 0.2, 0.2), seed=0)
        train_obj = {r["object_id"] for r in splits["train"]}
        val_obj = {r["object_id"] for r in splits["val"]}
        test_obj = {r["object_id"] for r in splits["test"]}
        self.assertFalse(train_obj & val_obj)
        self.assertFalse(train_obj & test_obj)
        self.assertFalse(val_obj & test_obj)
        self.assertEqual(sum(map(len, splits.values())), 20)

    def test_synthetic_dataset_item(self):
        ds = SyntheticGlueDataset("roi", img_size=(160, 160), length=2, seed=0)
        item = ds[0]
        self.assertEqual(tuple(item["image"].shape), (1, 160, 160))
        self.assertEqual(item["boxes"].shape[1], 4)
        ds_ref = SyntheticGlueDataset("reflection", img_size=(160, 160), length=7, seed=0)
        self.assertEqual(sum(1 for i in range(7) if ds_ref[i]["boxes"].shape[0] > 0), 3)


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestIntegration(unittest.TestCase):
    """与 algo.models 的接口集成：权重存在时可用，缺失时降级。"""

    def test_factory_degrades_without_weights(self):
        from algo.config import load_config
        from algo.models import HeuristicRoILocator, NullReflector, build_reflector, build_roi_locator

        cfg = load_config()
        cfg["locate"]["type"] = "rpn"
        cfg["locate"]["weights"] = None
        cfg["reflect"]["type"] = "yolo5fl"
        cfg["reflect"]["weights"] = None
        locator, reason = build_roi_locator(cfg)
        self.assertIsInstance(locator, HeuristicRoILocator)
        self.assertIn("RPN", reason or "")
        reflector, reason = build_reflector(cfg)
        self.assertIsInstance(reflector, NullReflector)
        self.assertIn("YOLO", reason or "")

    def test_rpn_locator_predicts_boxes(self):
        from algo.config import load_config
        from algo.models import RPNLocator

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rpn.pt"
            RPN(RPNConfig(width=8)).save(path)
            cfg = load_config()
            cfg["locate"]["weights"] = str(path)
            cfg["locate"]["input_size"] = 160
            cfg["locate"]["score_thr"] = 0.0
            locator = RPNLocator(cfg)
            image = np.zeros((120, 160), dtype=np.uint8)
            boxes = locator.predict(image)
        self.assertIsInstance(boxes, list)
        for box in boxes:
            self.assertLessEqual(box.x1, box.x2)
            self.assertLessEqual(box.y1, box.y2)

    def test_yolo_locator_predicts(self):
        from algo.config import load_config
        from algo.models import YOLO5FLDetector

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yolo.pt"
            YOLO5FL(YOLOConfig(num_levels=3)).save(path)
            cfg = load_config()
            cfg["reflect"]["weights"] = str(path)
            cfg["reflect"]["input_size"] = 160
            detector = YOLO5FLDetector(cfg)
            image = np.zeros((120, 160), dtype=np.uint8)
            result = detector.predict(image)
        self.assertFalse(result.has_reflection)
        self.assertEqual(result.boxes, [])


if __name__ == "__main__":
    unittest.main()
