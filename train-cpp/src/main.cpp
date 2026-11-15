// 训练入口：RPN / YOLO5FL（libtorch，批训练 + 计时 + 吞吐输出）。
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "data.h"
#include "rpn.h"
#include "yolo.h"

static std::vector<int64_t> level_strides(int levels) {
    return levels == 5 ? std::vector<int64_t>{32, 16, 8, 4, 2} : std::vector<int64_t>{32, 16, 8};
}

struct YoloLossOut {
    torch::Tensor obj, cls, box, total;
};

// 目标分配 + 损失（对应 Python yolo_targets / yolo_loss 的批处理版本）
static YoloLossOut yolo_loss(const std::vector<torch::Tensor>& outs,
                             const std::vector<torch::Tensor>& gts, int img, int levels,
                             const std::vector<std::vector<std::vector<float>>>& anchors) {
    auto strides = level_strides(levels);
    auto obj_sum = torch::zeros({});
    auto cls_sum = torch::zeros({});
    auto box_sum = torch::zeros({});
    int64_t n_pos = 0;
    const int64_t batch = outs[0].size(0);

    for (int64_t b = 0; b < batch; ++b) {
        auto gt = gts[static_cast<size_t>(b)];
        std::vector<std::pair<int, int>> assign;  // (level, anchor)
        for (int64_t g = 0; g < gt.size(0); ++g) {
            float gw = gt[g][2].item<float>() - gt[g][0].item<float>();
            float gh = gt[g][3].item<float>() - gt[g][1].item<float>();
            int best_l = 0, best_a = 0;
            float best = -1.0f;
            for (int l = 0; l < levels; ++l) {
                for (int a = 0; a < 3; ++a) {
                    float aw = anchors[static_cast<size_t>(l)][a][0];
                    float ah = anchors[static_cast<size_t>(l)][a][1];
                    float iw = std::min(gw, aw), ih = std::min(gh, ah);
                    float iou = (iw * ih) / (gw * gh + aw * ah - iw * ih + 1e-7f);
                    if (iou > best) { best = iou; best_l = l; best_a = a; }
                }
            }
            assign.emplace_back(best_l, best_a);
        }

        for (int l = 0; l < levels; ++l) {
            auto out = outs[static_cast<size_t>(l)][b];  // (3,HW,6)
            const int64_t hw = out.size(1);
            const int64_t stride = strides[static_cast<size_t>(l)];
            const int64_t side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(hw))));
            auto obj_t = torch::zeros({3, hw});
            auto mask = torch::zeros({3, hw}, torch::kBool);
            auto cls_t = torch::zeros({3, hw, 1});
            auto box_t = torch::zeros({3, hw, 4});

            for (size_t g = 0; g < assign.size(); ++g) {
                if (assign[g].first != l) continue;
                int a = assign[g].second;
                float cx = ((gt[g][0].item<float>() + gt[g][2].item<float>()) / 2) / stride;
                float cy = ((gt[g][1].item<float>() + gt[g][3].item<float>()) / 2) / stride;
                int gi = std::min<int>(static_cast<int>(cx), static_cast<int>(side) - 1);
                int gj = std::min<int>(static_cast<int>(cy), static_cast<int>(side) - 1);
                int64_t idx = static_cast<int64_t>(gj) * side + gi;
                obj_t[a][idx] = 1.0;
                mask[a][idx] = true;
                cls_t[a][idx][0] = 1.0;
                box_t[a][idx][0] = cx - gi;
                box_t[a][idx][1] = cy - gj;
                float aw = anchors[static_cast<size_t>(l)][a][0];
                float ah = anchors[static_cast<size_t>(l)][a][1];
                float gw = gt[g][2].item<float>() - gt[g][0].item<float>();
                float gh = gt[g][3].item<float>() - gt[g][1].item<float>();
                box_t[a][idx][2] = std::log(gw / aw);
                box_t[a][idx][3] = std::log(gh / ah);
                ++n_pos;
            }

            auto obj_logit = out.index({torch::indexing::Slice(), torch::indexing::Slice(), 4});
            auto cls_logit = out.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                        torch::indexing::Slice(5, 6)});
            auto box_pred = out.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                       torch::indexing::Slice(0, 4)});
            float pos = obj_t.sum().item<float>();
            float neg = static_cast<float>(obj_t.numel()) - pos;
            auto pw = torch::tensor(std::min(50.0f, std::max(1.0f, neg / std::max(1.0f, pos))));
            obj_sum = obj_sum + torch::binary_cross_entropy_with_logits(
                                      obj_logit, obj_t, {}, pw, torch::Reduction::Sum);

            auto idxs = mask.nonzero();  // (P,2): anchor, cell
            if (idxs.size(0) > 0) {
                auto a_idx = idxs.select(1, 0);
                auto cell = idxs.select(1, 1);
                auto gi = (cell % side).to(torch::kFloat32);
                auto gj = (cell / side).to(torch::kFloat32);
                auto pred4 = box_pred.index({a_idx, cell});
                auto tgt4 = box_t.index({a_idx, cell});
                auto cls_sel = cls_logit.index({a_idx, cell}).reshape({-1});
                auto cls_gt = cls_t.index({a_idx, cell}).reshape({-1});
                cls_sum = cls_sum + torch::binary_cross_entropy_with_logits(cls_sel, cls_gt);

                std::vector<float> aws(3), ahs(3);
                for (int a = 0; a < 3; ++a) {
                    aws[static_cast<size_t>(a)] = anchors[static_cast<size_t>(l)][a][0];
                    ahs[static_cast<size_t>(a)] = anchors[static_cast<size_t>(l)][a][1];
                }
                auto aw = torch::tensor(aws).index({a_idx});
                auto ah = torch::tensor(ahs).index({a_idx});
                auto pred_cx = (torch::sigmoid(pred4.select(1, 0)) + gi) * stride;
                auto pred_cy = (torch::sigmoid(pred4.select(1, 1)) + gj) * stride;
                auto pred_w = torch::exp(pred4.select(1, 2).clamp(-8, 8)) * aw;
                auto pred_h = torch::exp(pred4.select(1, 3).clamp(-8, 8)) * ah;
                auto pred_box = torch::stack(
                    {pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_cx + pred_w / 2,
                     pred_cy + pred_h / 2}, 1);
                auto tgt_cx = (tgt4.select(1, 0) + gi) * stride;
                auto tgt_cy = (tgt4.select(1, 1) + gj) * stride;
                auto tgt_w = torch::exp(tgt4.select(1, 2)) * aw;
                auto tgt_h = torch::exp(tgt4.select(1, 3)) * ah;
                auto tgt_box = torch::stack(
                    {tgt_cx - tgt_w / 2, tgt_cy - tgt_h / 2, tgt_cx + tgt_w / 2,
                     tgt_cy + tgt_h / 2}, 1);
                box_sum = box_sum + (1.0 - ciou_sim(pred_box, tgt_box)).mean();
            }
        }
    }
    auto denom = std::max<int64_t>(1, n_pos);
    auto total = (obj_sum + cls_sum + box_sum) / denom;
    return {obj_sum / denom, cls_sum / denom, box_sum / denom, total};
}

static void print_anchor(const torch::Tensor& anchors) {
    auto a = anchors.to(torch::kCPU).contiguous();
    std::string s = "[rpn] kmeans anchors: ";
    for (int i = 0; i < std::min<int64_t>(a.size(0), 9); ++i) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "(%.0f,%.0f) ", a[i][0].item<float>(), a[i][1].item<float>());
        s += buf;
    }
    log_line("%s\n", s.c_str());
}

int main(int argc, char** argv) {
    TrainCfg cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--task") cfg.task = next();
        else if (a == "--epochs") cfg.epochs = std::stoi(next());
        else if (a == "--batch") cfg.batch = std::stoi(next());
        else if (a == "--img") cfg.img = std::stoi(next());
        else if (a == "--width") cfg.width = std::stoi(next());
        else if (a == "--levels") cfg.levels = std::stoi(next());
        else if (a == "--no-spp") cfg.spp = false;
        else if (a == "--lr") cfg.lr = std::stod(next());
        else if (a == "--threads") cfg.threads = std::stoi(next());
        else if (a == "--train-n") cfg.train_n = std::stoi(next());
        else if (a == "--seed") cfg.seed = std::stoll(next());
        else if (a == "--bench") cfg.epochs = 1;
        else if (a == "--help" || a == "-h") {
            std::printf("用法: train_cpp --task rpn|yolo [--epochs N --batch B --img S --threads T "
                        "--train-n N --lr L --levels 3|5 --no-spp --bench]\n");
            return 0;
        }
    }
    if (cfg.threads > 0) torch::set_num_threads(cfg.threads);
    torch::manual_seed(static_cast<uint64_t>(cfg.seed));
    const bool reflection = cfg.task == "yolo";

    log_line("[cfg] task=%s img=%d batch=%d epochs=%d lr=%.4g threads=%d\n", cfg.task.c_str(),
             cfg.img, cfg.batch, cfg.epochs, cfg.lr, cfg.threads);

    auto train = make_dataset(cfg.train_n, cfg.img, reflection, cfg.seed);
    std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
    std::vector<int> order(static_cast<size_t>(cfg.train_n));
    for (int i = 0; i < cfg.train_n; ++i) order[static_cast<size_t>(i)] = i;

    Timer total_timer;
    int64_t seen = 0;

    if (cfg.task == "rpn") {
        std::vector<torch::Tensor> whs;
        for (auto& s : train)
            for (int64_t i = 0; i < s.boxes.size(0); ++i) {
                auto b = s.boxes[i];
                whs.push_back(torch::stack({b[2] - b[0], b[3] - b[1]}));
            }
        auto anchors = wh_kmeans(torch::stack(whs), 9, cfg.seed);
        print_anchor(anchors);
        RPN model(1, cfg.width, cfg.img, anchors);
        auto params = model.parameters();
        int64_t nparam = 0;
        for (auto& p : params) nparam += p.numel();
        log_line("[rpn] params=%.2fM\n", nparam / 1e6);
        torch::optim::AdamW opt(params, torch::optim::AdamWOptions(cfg.lr));
        for (int ep = 0; ep < cfg.epochs; ++ep) {
            std::shuffle(order.begin(), order.end(), rng);
            Timer et;
            double loss_sum = 0;
            int steps = 0;
            for (int i = 0; i < cfg.train_n; i += cfg.batch) {
                int64_t n = std::min<int>(cfg.batch, cfg.train_n - i);
                std::vector<int> idx(order.begin() + i, order.begin() + i + n);
                auto images = stack_images(train, idx);
                std::vector<torch::Tensor> gts;
                for (int j : idx) gts.push_back(train[static_cast<size_t>(j)].boxes);
                auto out = model.forward_out(images);
                auto losses = rpn_loss(std::get<0>(out), std::get<1>(out), std::get<2>(out), gts);
                opt.zero_grad();
                losses.total.backward();
                torch::nn::utils::clip_grad_norm_(model.parameters(), 10.0);
                opt.step();
                loss_sum += losses.total.item<double>();
                ++steps;
                seen += n;
            }
            double ms = et.ms();
            log_line("[rpn] epoch %d/%d loss=%.4f %.0f ms (%.0f samples/s)\n", ep + 1,
                     cfg.epochs, loss_sum / std::max(1, steps), ms,
                     cfg.train_n / (ms / 1000.0));
        }
    } else {
        YOLO model(1, cfg.levels, cfg.spp, cfg.img);
        auto params = model->parameters();
        int64_t nparam = 0;
        for (auto& p : params) nparam += p.numel();
        log_line("[yolo] levels=%d spp=%d params=%.2fM\n", cfg.levels, cfg.spp ? 1 : 0,
                 nparam / 1e6);
        torch::optim::AdamW opt(params, torch::optim::AdamWOptions(cfg.lr));
        for (int ep = 0; ep < cfg.epochs; ++ep) {
            std::shuffle(order.begin(), order.end(), rng);
            Timer et;
            double loss_sum = 0;
            int steps = 0;
            for (int i = 0; i < cfg.train_n; i += cfg.batch) {
                int64_t n = std::min<int>(cfg.batch, cfg.train_n - i);
                std::vector<int> idx(order.begin() + i, order.begin() + i + n);
                auto images = stack_images(train, idx);
                std::vector<torch::Tensor> gts;
                for (int j : idx) gts.push_back(train[static_cast<size_t>(j)].boxes);
                auto outs = model->forward(images);
                auto losses = yolo_loss(outs, gts, cfg.img, cfg.levels, model->anchors_);
                opt.zero_grad();
                losses.total.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 10.0);
                opt.step();
                loss_sum += losses.total.item<double>();
                ++steps;
                seen += n;
            }
            double ms = et.ms();
            log_line("[yolo] epoch %d/%d loss=%.4f %.0f ms (%.0f samples/s)\n", ep + 1,
                     cfg.epochs, loss_sum / std::max(1, steps), ms,
                     cfg.train_n / (ms / 1000.0));
        }
    }
    double total_ms = total_timer.ms();
    log_line("[done] %.0f ms, %.0f samples/s, %.1f ms/sample\n", total_ms,
             seen / (total_ms / 1000.0), total_ms / std::max<int64_t>(1, seen));
    return 0;
}
