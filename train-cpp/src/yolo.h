// YOLOv3-5FL + SPP（论文 3.3）：5 层输出、SPP、15 个先验框、CIoU 回归。
#pragma once

#include <torch/torch.h>

#include <vector>

#include "common.h"

inline torch::Tensor ciou_sim(const torch::Tensor& a, const torch::Tensor& b) {
    auto ax1 = a.select(1, 0), ay1 = a.select(1, 1), ax2 = a.select(1, 2), ay2 = a.select(1, 3);
    auto bx1 = b.select(1, 0), by1 = b.select(1, 1), bx2 = b.select(1, 2), by2 = b.select(1, 3);
    auto iw = (torch::min(ax2, bx2) - torch::max(ax1, bx1)).clamp_min(0);
    auto ih = (torch::min(ay2, by2) - torch::max(ay1, by1)).clamp_min(0);
    auto inter = iw * ih;
    auto wa = (ax2 - ax1).clamp_min(1e-7), ha = (ay2 - ay1).clamp_min(1e-7);
    auto wb = (bx2 - bx1).clamp_min(1e-7), hb = (by2 - by1).clamp_min(1e-7);
    auto iou = inter / (wa * ha + wb * hb - inter).clamp_min(1e-7);
    auto cw = torch::max(ax2, bx2) - torch::min(ax1, bx1);
    auto ch = torch::max(ay2, by2) - torch::min(ay1, by1);
    auto rho2 = torch::pow((ax1 + ax2 - bx1 - bx2) / 2, 2) + torch::pow((ay1 + ay2 - by1 - by2) / 2, 2);
    auto v = (4 / (M_PI * M_PI)) * torch::pow(torch::atan(wa / ha) - torch::atan(wb / hb), 2);
    auto alpha = v / (1 - iou + v).clamp_min(1e-7);
    return iou - rho2 / (cw * cw + ch * ch).clamp_min(1e-7) - alpha * v;
}

// Conv + BN + SiLU（CBL 单元）
struct ConvBlockImpl : torch::nn::Module {
    ConvBlockImpl(int in, int out, int k = 3, int s = 1) {
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in, out, k).stride(s).padding(k / 2).bias(false)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out));
    }
    torch::Tensor forward(torch::Tensor x) { return torch::silu(bn->forward(conv->forward(x))); }
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(ConvBlock);

struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int ch) {
        cv1 = register_module("cv1", ConvBlock(ch, ch / 2, 1));
        cv2 = register_module("cv2", ConvBlock(ch / 2, ch, 3));
    }
    torch::Tensor forward(torch::Tensor x) { return x + cv2->forward(cv1->forward(x)); }
    ConvBlock cv1{nullptr}, cv2{nullptr};
};
TORCH_MODULE(ResBlock);

struct SPPImpl : torch::nn::Module {
    SPPImpl(int in, int out) : mid_(in / 2) {
        cv1 = register_module("cv1", ConvBlock(in, mid_, 1));
        cv2 = register_module("cv2", ConvBlock(mid_ * 4, out, 3));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = cv1->forward(x);
        auto h = x.size(2), w = x.size(3);
        std::vector<torch::Tensor> feats = {x};
        for (int bins : {4, 2, 1}) {
            auto p = std::get<0>(torch::adaptive_max_pool2d(x, std::vector<int64_t>{bins, bins}));
            feats.push_back(torch::nn::functional::interpolate(
                p, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{h, w})
                       .mode(torch::kNearest)));
        }
        return cv2->forward(torch::cat(feats, 1));
    }
    int mid_;
    ConvBlock cv1{nullptr}, cv2{nullptr};
};
TORCH_MODULE(SPP);

inline const float (*anchor_table())[3][2] {
    static const float tbl[5][3][2] = {
        {{200, 60}, {280, 80}, {360, 100}},
        {{110, 40}, {160, 56}, {210, 70}},
        {{60, 26}, {90, 36}, {120, 48}},
        {{40, 20}, {60, 28}, {84, 38}},
        {{28, 16}, {40, 22}, {56, 30}},
    };
    return tbl;
}

inline std::vector<std::vector<std::vector<float>>> default_anchors(int levels) {
    const float (*tbl)[3][2] = anchor_table();
    int base = 5 - levels;
    std::vector<std::vector<std::vector<float>>> out;
    for (int l = 0; l < levels; ++l) {
        std::vector<std::vector<float>> level;
        for (int a = 0; a < 3; ++a) level.push_back({tbl[base + l][a][0], tbl[base + l][a][1]});
        out.push_back(level);
    }
    return out;
}

struct YOLOImpl : torch::nn::Module {
    YOLOImpl(int in_ch, int levels, bool spp, int /*img*/) : levels_(levels), spp_(spp) {
        stem = register_module("stem", ConvBlock(in_ch, 32, 3, 2));
        s1a = register_module("s1a", ConvBlock(32, 64, 3, 2));
        s1b = register_module("s1b", ResBlock(64));
        s1c = register_module("s1c", ResBlock(64));
        s2a = register_module("s2a", ConvBlock(64, 128, 3, 2));
        s2b = register_module("s2b", ResBlock(128));
        s2c = register_module("s2c", ResBlock(128));
        s3a = register_module("s3a", ConvBlock(128, 256, 3, 2));
        s3b = register_module("s3b", ResBlock(256));
        s3c = register_module("s3c", ResBlock(256));
        s4a = register_module("s4a", ConvBlock(256, 512, 3, 2));
        s4b = register_module("s4b", ResBlock(512));
        s4c = register_module("s4c", ResBlock(512));
        if (spp) {
            spp_mod = register_module("spp", SPP(512, 128));
        } else {
            spp_alt = register_module("spp_alt", ConvBlock(512, 128, 1));
        }
        lat4 = register_module("lat4", ConvBlock(256, 128, 1));
        lat3 = register_module("lat3", ConvBlock(128, 128, 1));
        neck5 = register_module("neck5", ConvBlock(128, 128, 3));
        neck4 = register_module("neck4", ConvBlock(128, 128, 3));
        neck3 = register_module("neck3", ConvBlock(128, 128, 3));
        neck2 = register_module("neck2", ConvBlock(128, 64, 3));
        neck1 = register_module("neck1", ConvBlock(64, 64, 3));
        h5 = register_module("h5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 18, 1)));
        h4 = register_module("h4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 18, 1)));
        h3 = register_module("h3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 18, 1)));
        h2 = register_module("h2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 18, 1)));
        h1 = register_module("h1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 18, 1)));
        anchors_ = default_anchors(levels);
    }

    static torch::Tensor up2(torch::Tensor x, int64_t target_h, int64_t target_w) {
        return torch::nn::functional::interpolate(
            x, torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<int64_t>{target_h, target_w})
                   .mode(torch::kNearest));
    }

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        auto x0 = stem->forward(x);
        auto c3 = s2c->forward(s2b->forward(s2a->forward(s1c->forward(s1b->forward(s1a->forward(x0))))));  // /8
        auto c4 = s3c->forward(s3b->forward(s3a->forward(c3)));  // /16
        auto c5 = s4c->forward(s4b->forward(s4a->forward(c4)));  // /32
        auto p5 = neck5->forward(spp_ ? spp_mod->forward(c5) : spp_alt->forward(c5));
        auto p4 = neck4->forward(lat4->forward(c4) + up2(p5, c4.size(2), c4.size(3)));
        auto p3 = neck3->forward(lat3->forward(c3) + up2(p4, c3.size(2), c3.size(3)));
        std::vector<torch::Tensor> feats = {p5, p4, p3};
        std::vector<torch::nn::Conv2d> heads = {h5, h4, h3};
        if (levels_ == 5) {
            auto p2 = neck2->forward(up2(p3, p3.size(2) * 2, p3.size(3) * 2));
            auto p1 = neck1->forward(up2(p2, p2.size(2) * 2, p2.size(3) * 2));
            feats.push_back(p2);
            feats.push_back(p1);
            heads.push_back(h2);
            heads.push_back(h1);
        }
        std::vector<torch::Tensor> out;
        for (size_t i = 0; i < feats.size(); ++i) {
            auto o = heads[i]->forward(feats[i]);
            auto b = o.size(0), h = o.size(2), w = o.size(3);
            o = o.view({b, 3, 6, h, w}).permute({0, 1, 3, 4, 2}).reshape({b, 3, h * w, 6});
            out.push_back(o);
        }
        return out;
    }

    std::vector<std::vector<std::vector<float>>> anchors_;
    int levels_;
    bool spp_;
    ConvBlock stem{nullptr}, s1a{nullptr}, s2a{nullptr}, s3a{nullptr}, s4a{nullptr},
        spp_alt{nullptr}, lat4{nullptr}, lat3{nullptr}, neck5{nullptr}, neck4{nullptr},
        neck3{nullptr}, neck2{nullptr}, neck1{nullptr};
    ResBlock s1b{nullptr}, s1c{nullptr}, s2b{nullptr}, s2c{nullptr}, s3b{nullptr}, s3c{nullptr},
        s4b{nullptr}, s4c{nullptr};
    SPP spp_mod{nullptr};
    torch::nn::Conv2d h1{nullptr}, h2{nullptr}, h3{nullptr}, h4{nullptr}, h5{nullptr};
};
TORCH_MODULE(YOLO);
