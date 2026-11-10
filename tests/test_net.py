"""B1/B2 模型测试（需要 torch；缺失时整组跳过）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    import torch

    from algo.data import SyntheticGlueDataset, letterbox, scale_boxes
    from algo.net.common import ciou_nms_torch, giou_diou_ciou, make_anchors, nms_torch
    from algo.net.rpn import RPN, ProposalConfig, decode_boxes, encode_boxes, propose, rpn_loss
    from algo.net.yolo import YOLO5FL, decode_yolo, yolo_loss, yolo_targets


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
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["loss"]), 0.0)

    def test_propose_returns_boxes(self):
        x = torch.zeros(1, 1, 160, 160)
        cls_logits, reg, anchors, _ = self.model(x)
        boxes, scores = propose(cls_logits[0], reg[0], anchors, (160, 160),
                                ProposalConfig(nms_type="ciou_nms", nms_thr=0.5))
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(boxes.shape[0], scores.shape[0])
        if boxes.numel():
            self.assertTrue(float(boxes[:, 0].min()) >= 0.0)
            self.assertTrue(float(boxes[:, 2].max()) <= 160.0)

    def test_anchor_count(self):
        anchors = make_anchors(4, 4, 16, (8, 16, 32), (0.5, 1.0, 2.0))
        self.assertEqual(anchors.shape, (4 * 4 * 9, 4))


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestYOLO(unittest.TestCase):
    def test_forward_5_levels(self):
        model = YOLO5FL(num_levels=5, use_spp=True).eval()
        with torch.no_grad():
            outs = model(torch.zeros(1, 1, 320, 320))
        self.assertEqual(len(outs), 5)
        sizes = [o.shape[3] for o in outs]  # (B,A,HW,5+nc)
        self.assertEqual(sizes, [100, 400, 1600, 6400, 25600])

    def test_forward_3_levels(self):
        model = YOLO5FL(num_levels=3, use_spp=False).eval()
        with torch.no_grad():
            outs = model(torch.zeros(1, 1, 320, 320))
        self.assertEqual(len(outs), 3)
        self.assertEqual([o.shape[3] for o in outs], [100, 400, 1600])

    def test_loss_finite_and_targets(self):
        model = YOLO5FL(num_levels=3, use_spp=True)
        x = torch.zeros(1, 1, 320, 320)
        outs = model(x)
        gt = torch.tensor([[60.0, 120, 260, 160]])
        targets = yolo_targets(outs, gt, (320, 320), anchors=model.anchors)
        losses = yolo_loss(outs, targets, (320, 320), anchors=model.anchors)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["loss"]), 0.0)
        self.assertEqual(sum(int(t["mask"].sum()) for t in targets), 1)

    def test_decode_returns_boxes(self):
        model = YOLO5FL(num_levels=3).eval()
        with torch.no_grad():
            outs = model(torch.zeros(1, 1, 320, 320))
            decoded = decode_yolo(outs, (320, 320), anchors=model.anchors, conf_thr=0.0)
        boxes, scores = decoded[0]
        self.assertEqual(boxes.shape[1], 4)


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestNZms(unittest.TestCase):
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
            self.assertTrue(torch.isfinite(giou_diou_ciou(a, b, mode=mode)))


@unittest.skipUnless(HAS_TORCH, "需要 torch（可选依赖）")
class TestData(unittest.TestCase):
    def test_letterbox_and_scale(self):
        import numpy as np

        img = np.zeros((320, 480), dtype=np.uint8)
        canvas, scale, pad = letterbox(img, (320, 320))
        self.assertEqual(canvas.shape, (320, 320))
        self.assertAlmostEqual(scale, 320 / 480, places=5)
        boxes = np.array([[60.0, 100, 420, 200]], dtype=np.float32)
        mapped = scale_boxes(boxes, scale, pad)
        self.assertAlmostEqual(float(mapped[0, 0]), 60 * scale + pad[0], places=4)

    def test_synthetic_dataset_item(self):
        ds = SyntheticGlueDataset("roi", img_size=(160, 160), length=2, seed=0)
        item = ds[0]
        self.assertEqual(tuple(item["image"].shape), (1, 160, 160))
        self.assertEqual(item["boxes"].shape[1], 4)
        ds_ref = SyntheticGlueDataset("reflection", img_size=(160, 160), length=5, seed=0)
        self.assertTrue(any(ds_ref[i]["boxes"].shape[0] > 0 for i in range(5)))


if __name__ == "__main__":
    unittest.main()
