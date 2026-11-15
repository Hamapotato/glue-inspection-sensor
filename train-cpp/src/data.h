// 合成训练数据：与 algo/synth.py 的"胶条/断胶/反光"场景对齐（灰度、归一化到 0..1）。
#pragma once

#include <torch/torch.h>

#include <random>
#include <vector>

struct Sample {
    torch::Tensor image;  // (1,H,W) float32
    torch::Tensor boxes;  // (N,4) float32, 像素坐标
};

inline void draw_strip(torch::Tensor& img, int x0, int x1, int y0, int y1, float v) {
    img.index_put_({torch::indexing::Slice(), torch::indexing::Slice(y0, y1),
                    torch::indexing::Slice(x0, x1)},
                   torch::tensor(v));
}

inline Sample make_sample(int img, bool reflection, std::mt19937& rng,
                          int kind) {  // kind: 0 正常 1 断胶 2 反光
    auto t = torch::rand({1, img, img}) * 0.12f + 0.04f;
    int strip_h = std::max(6, static_cast<int>(img * 0.09f));
    int y0 = img / 2 - strip_h / 2;
    int x0 = img / 12, x1 = img - img / 12;
    std::vector<float> boxes;
    if (kind == 1) {
        int mid = (x0 + x1) / 2;
        int gap = img / 20;
        draw_strip(t, x0, mid - gap, y0, y0 + strip_h, 0.9f);
        draw_strip(t, mid + gap, x1, y0, y0 + strip_h, 0.9f);
        boxes = {float(x0), float(y0), float(mid - gap), float(y0 + strip_h),
                 float(mid + gap), float(y0), float(x1), float(y0 + strip_h)};
    } else {
        draw_strip(t, x0, x1, y0, y0 + strip_h, 0.9f);
        boxes = {float(x0), float(y0), float(x1), float(y0 + strip_h)};
    }
    if (reflection && kind == 2) {
        int rw = std::max(4, img / 22), rh = std::max(3, img / 40);
        int cx = img / 2, cy = img / 2;
        draw_strip(t, cx - rw, cx + rw, cy - rh, cy + rh, 1.0f);
        boxes = {float(cx - rw), float(cy - rh), float(cx + rw), float(cy + rh)};
    }
    return {t, torch::tensor(boxes).reshape({-1, 4})};
}

inline std::vector<Sample> make_dataset(int n, int img, bool reflection, int64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::vector<Sample> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        int kind = reflection ? (i % 3 == 2 ? 2 : (i % 3 == 0 ? 0 : 1)) : (i % 4 == 3 ? 1 : 0);
        out.push_back(make_sample(img, reflection, rng, kind));
    }
    return out;
}

inline torch::Tensor stack_images(const std::vector<Sample>& ds, const std::vector<int>& idx) {
    std::vector<torch::Tensor> xs;
    xs.reserve(idx.size());
    for (int i : idx) xs.push_back(ds[i].image);
    return torch::stack(xs, 0);
}
