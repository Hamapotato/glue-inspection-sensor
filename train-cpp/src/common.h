// 公共：计时、训练配置、日志。
#pragma once

#include <torch/torch.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <string>

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void reset() { t0 = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

struct TrainCfg {
    std::string task = "rpn";   // rpn | yolo
    int epochs = 30;
    int batch = 16;
    int img = 160;
    int width = 16;             // RPN 骨干宽度
    int levels = 5;             // YOLO 输出层数
    bool spp = true;
    double lr = 2e-3;
    int threads = 0;            // 0 = torch 默认
    int train_n = 128;
    int val_n = 32;
    int64_t seed = 0;
    int log_every = 1;
};

inline void log_line(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::fflush(stdout);
}
