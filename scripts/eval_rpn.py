"""评估 RPN：RoI 定位平均 IoU（M1）+ 候选框筛选方式对比（表 3-2 口径）。

用法：
    py -3 scripts/eval_rpn.py --weights outputs/train_rpn/rpn.pt --synthetic
    py -3 scripts/eval_rpn.py --weights ... --nms ciou_nms --json outputs/eval_rpn.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from algo.data import SyntheticGlueDataset  # noqa: E402
from algo.net.rpn import RPN, ProposalConfig  # noqa: E402
from algo.train import evaluate_rpn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 RPN")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--nms", default="nms", choices=["nms", "ciou_nms"])
    parser.add_argument("--nms-thr", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    model = RPN()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    img_size = (args.img_size, args.img_size)
    ds = SyntheticGlueDataset("roi", img_size=img_size, length=args.length, seed=777)
    prop = ProposalConfig(nms_type=args.nms, nms_thr=args.nms_thr, score_thr=args.score_thr)
    metrics = evaluate_rpn(model, ds, proposal_cfg=prop, device_name=args.device)

    result = {"mean_iou": metrics["mean_iou"], "n": metrics["n"],
              "nms": args.nms, "nms_thr": args.nms_thr, "weights": args.weights}
    print(f"mean IoU = {result['mean_iou']:.4f} (n={result['n']}, nms={args.nms}, thr={args.nms_thr})")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
