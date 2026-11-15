// RPN（论文 3.1）：轻量骨干 + 双分支头 + K-means 先验框 + DIoU 回归损失。
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <random>
#include <vector>

#include "common.h"
#include "data.h"

inline torch::Tensor wh_iou_matrix(const torch::Tensor& a, const torch::Tensor& b) {
    // a (N,2), b (M,2) -> (N,M) 宽高 IoU
    auto aw = a.select(1, 0).unsqueeze(1), ah = a.select(1, 1).unsqueeze(1);
    auto bw = b.select(1, 0).unsqueeze(0), bh = b.select(1, 1).unsqueeze(0);
    auto iw = torch::min(aw, bw), ih = torch::min(ah, bh);
    auto inter = iw * ih;
    return inter / (aw * ah + bw * bh - inter + 1e-7);
}

// K-means 生成先验框（论文 3.1.2）
inline torch::Tensor wh_kmeans(torch::Tensor wh, int k, int64_t seed) {
    wh = wh.to(torch::kCPU, torch::kFloat32);
    const int64_t n = wh.size(0);
    k = static_cast<int>(std::min<int64_t>(k, n));
    torch::manual_seed(seed);
    auto idx = torch::randperm(n).slice(0, 0, k);
    auto centers = wh.index_select(0, idx).clone();
    for (int it = 0; it < 25; ++it) {
        auto sim = wh_iou_matrix(wh, centers);        // (n,k)
        auto assign = sim.argmax(1);                  // 1 - IoU 距离最小
        auto next = centers.clone();
        for (int c = 0; c < k; ++c) {
            auto mask = (assign == c);
            if (mask.sum().item<int64_t>() > 0) next[c] = wh.index({mask}).mean(0);
        }
        if (torch::allclose(next, centers, 1e-5, 1e-6)) { centers = next; break; }
        centers = next;
    }
    // 面积升序，保证 anchor 顺序稳定
    auto area = centers.select(1, 0) * centers.select(1, 1);
    auto order = torch::argsort(area);
    return centers.index_select(0, order);
}

inline torch::Tensor make_anchors(int feat_h, int feat_w, int stride, const torch::Tensor& wh) {
    const int64_t a = wh.size(0);
    auto xs = (torch::arange(feat_w, torch::kFloat32) + 0.5f) * stride;
    auto ys = (torch::arange(feat_h, torch::kFloat32) + 0.5f) * stride;
    auto grid = torch::meshgrid({ys, xs}, "ij");
    auto gy = grid[0].reshape(-1), gx = grid[1].reshape(-1);
    auto cx = gx.unsqueeze(1).expand({-1, a}).reshape(-1);
    auto cy = gy.unsqueeze(1).expand({-1, a}).reshape(-1);
    auto aw = wh.select(1, 0).unsqueeze(0).expand({gx.size(0), -1}).reshape(-1);
    auto ah = wh.select(1, 1).unsqueeze(0).expand({gx.size(0), -1}).reshape(-1);
    return torch::stack({cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2}, 1);
}

inline torch::Tensor box_iou_1n(const torch::Tensor& boxes, const torch::Tensor& g) {
    auto b = boxes;
    auto gx1 = g[0].item<float>(), gy1 = g[1].item<float>();
    auto gx2 = g[2].item<float>(), gy2 = g[3].item<float>();
    auto iw = (torch::min(b.select(1, 2), torch::tensor(gx2)) - torch::max(b.select(1, 0), torch::tensor(gx1))).clamp_min(0);
    auto ih = (torch::min(b.select(1, 3), torch::tensor(gy2)) - torch::max(b.select(1, 1), torch::tensor(gy1))).clamp_min(0);
    auto inter = iw * ih;
    auto area_b = (b.select(1, 2) - b.select(1, 0)) * (b.select(1, 3) - b.select(1, 1));
    auto area_g = (gx2 - gx1) * (gy2 - gy1);
    return inter / (area_b + area_g - inter + 1e-7);
}

inline torch::Tensor diou_sim(const torch::Tensor& a, const torch::Tensor& b) {
    auto ax1 = a.select(1, 0), ay1 = a.select(1, 1), ax2 = a.select(1, 2), ay2 = a.select(1, 3);
    auto bx1 = b.select(1, 0), by1 = b.select(1, 1), bx2 = b.select(1, 2), by2 = b.select(1, 3);
    auto iw = (torch::min(ax2, bx2) - torch::max(ax1, bx1)).clamp_min(0);
    auto ih = (torch::min(ay2, by2) - torch::max(ay1, by1)).clamp_min(0);
    auto inter = iw * ih;
    auto area_a = (ax2 - ax1) * (ay2 - ay1);
    auto area_b = (bx2 - bx1) * (by2 - by1);
    auto iou = inter / (area_a + area_b - inter).clamp_min(1e-7);
    auto cw = torch::max(ax2, bx2) - torch::min(ax1, bx1);
    auto ch = torch::max(ay2, by2) - torch::min(ay1, by1);
    auto rho2 = torch::pow((ax1 + ax2 - bx1 - bx2) / 2, 2) + torch::pow((ay1 + ay2 - by1 - by2) / 2, 2);
    return iou - rho2 / (cw * cw + ch * ch).clamp_min(1e-7);
}

inline torch::Tensor decode_boxes(const torch::Tensor& anchors, const torch::Tensor& d) {
    auto aw = (anchors.select(1, 2) - anchors.select(1, 0)).clamp_min(1e-7);
    auto ah = (anchors.select(1, 3) - anchors.select(1, 1)).clamp_min(1e-7);
    auto ax = (anchors.select(1, 0) + anchors.select(1, 2)) / 2;
    auto ay = (anchors.select(1, 1) + anchors.select(1, 3)) / 2;
    auto cx = d.select(1, 0) * aw + ax;
    auto cy = d.select(1, 1) * ah + ay;
    auto w = torch::exp(d.select(1, 2).clamp(-10, 10)) * aw;
    auto h = torch::exp(d.select(1, 3).clamp(-10, 10)) * ah;
    return torch::stack({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2}, 1);
}

struct RPN : torch::nn::Module {
    RPN(int in_ch, int width, int img, torch::Tensor wh)
        : stride_(16), img_(img), wh_(std::move(wh)) {
        const int c = width;
        b1_ = register_module("b1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, c, 3).stride(2).padding(1)),
            torch::nn::BatchNorm2d(c), torch::nn::SiLU()));
        b2_ = register_module("b2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c * 2, 3).stride(2).padding(1)),
            torch::nn::BatchNorm2d(c * 2), torch::nn::SiLU()));
        b3_ = register_module("b3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c * 2, c * 4, 3).stride(2).padding(1)),
            torch::nn::BatchNorm2d(c * 4), torch::nn::SiLU()));
        b4_ = register_module("b4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c * 4, c * 8, 3).stride(2).padding(1)),
            torch::nn::BatchNorm2d(c * 8), torch::nn::SiLU()));
        conv_ = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(c * 8, 256, 3).padding(1)));
        cls_ = register_module("cls", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 9, 1)));
        reg_ = register_module("reg", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 36, 1)));
        anchors_ = make_anchors(img_ / stride_, img_ / stride_, stride_, wh_);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_out(torch::Tensor x) {
        auto f = b4_->forward(b3_->forward(b2_->forward(b1_->forward(x))));
        f = torch::silu(conv_->forward(f));
        auto shape = f.sizes();
        const int64_t b = shape[0], h = shape[2], w = shape[3];
        auto cls = cls_->forward(f).permute({0, 2, 3, 1}).reshape({b, h * w * 9});
        auto reg = reg_->forward(f).permute({0, 2, 3, 1}).reshape({b, h * w * 9, 4});
        return {cls, reg, anchors_};
    }

    torch::nn::Sequential b1_{nullptr}, b2_{nullptr}, b3_{nullptr}, b4_{nullptr};
    torch::nn::Conv2d conv_{nullptr}, cls_{nullptr}, reg_{nullptr};
    torch::Tensor anchors_;
    int stride_, img_;
    torch::Tensor wh_;
};

struct RpnLoss {
    torch::Tensor cls, reg, total;
};

inline RpnLoss rpn_loss(const torch::Tensor& cls, const torch::Tensor& reg,
                        const torch::Tensor& anchors, const std::vector<torch::Tensor>& gts,
                        float pos_thr = 0.5f, float neg_thr = 0.3f, int samples = 256,
                        float pos_frac = 0.5f) {
    auto cls_loss = torch::zeros({});
    auto reg_loss = torch::zeros({});
    const int64_t batch = cls.size(0);
    for (int64_t b = 0; b < batch; ++b) {
        auto gt = gts[static_cast<size_t>(b)];
        auto logits = cls[b];
        if (gt.numel() == 0) {
            cls_loss = cls_loss + torch::binary_cross_entropy_with_logits(logits, torch::zeros_like(logits));
            continue;
        }
        std::vector<torch::Tensor> cols;
        for (int64_t g = 0; g < gt.size(0); ++g) cols.push_back(box_iou_1n(anchors, gt[g]).unsqueeze(1));
        auto ious = torch::cat(cols, 1);  // (N,G)
        auto best = ious.max(1);
        auto best_iou = std::get<0>(best), best_idx = std::get<1>(best);
        auto pos = best_iou >= pos_thr;
        auto neg = best_iou < neg_thr;
        auto per_gt = ious.argmax(0);  // 每个 GT 的最佳 anchor
        for (int64_t g = 0; g < gt.size(0); ++g) pos.index_put_({per_gt[g]}, true);

        auto pos_idx = pos.nonzero().reshape(-1);
        auto neg_idx = neg.nonzero().reshape(-1);
        const int64_t n_pos = pos_idx.size(0);
        int64_t num_pos = std::min<int64_t>(static_cast<int64_t>(pos_frac * samples), n_pos);
        int64_t num_neg = std::min<int64_t>(samples - num_pos, neg_idx.size(0));
        auto pos_sorted = best_iou.index_select(0, pos_idx).argsort(static_cast<int64_t>(0));
        pos_idx = pos_idx.index_select(0, pos_sorted).slice(0, 0, num_pos);
        if (num_neg < neg_idx.size(0))
            neg_idx = neg_idx.index_select(0, torch::randperm(neg_idx.size(0)).slice(0, 0, num_neg));
        auto sel = torch::cat({pos_idx, neg_idx});
        auto labels = torch::cat({torch::ones({pos_idx.size(0)}), torch::zeros({neg_idx.size(0)})});
        cls_loss = cls_loss + torch::binary_cross_entropy_with_logits(logits.index_select(0, sel), labels);

        if (pos_idx.size(0) > 0) {
            auto matched = gt.index_select(0, best_idx.index_select(0, pos_idx));
            auto pred = decode_boxes(anchors.index_select(0, pos_idx), reg[b].index_select(0, pos_idx));
            reg_loss = reg_loss + (1.0 - diou_sim(pred, matched)).mean();
        }
    }
    return {cls_loss / batch, reg_loss / batch, cls_loss / batch + reg_loss / batch};
}
