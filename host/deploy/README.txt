涂胶检测软件 · Ubuntu 板端部署包

包含两个静态二进制（musl，无 glibc 依赖）：
  bin/aarch64/glue-agent   # RK3588 等 ARM64 板子
  bin/x86_64/glue-agent    # x86_64 Ubuntu 机器

三步部署：
  1) sudo ./install.sh          # 安装到 /opt/glue，自动按 uname -m 选架构
  2) cd /opt/glue && sudo ./verify.sh   # 跑 8 张样例并打印判定统计
  3) sudo systemctl enable --now glue-agent   # 可选：常驻扫描 /opt/glue/samples

GUI（需要桌面环境，在板子上编译）：
  在仓库根目录执行：sudo ./deploy/build-gui-on-board.sh
  或手动：apt 安装 libx11-dev libgl1-mesa-dev 等后，cd host && cargo build --release -p glue-app

目录：
  configs/default.yaml   算法与质量门控配置（标准宽度、报警线、quality 阈值）
  samples/               8 张合成样例（正常/过窄/过宽/断胶/无胶/过暗/过曝/模糊）
  deploy/                systemd 单元与脚本（install/verify/build-gui-on-board）
  bin/                   按架构选择的静态二进制

结果说明：
  logs/verify.json 中每个文件含 verdict / width_mm / timings_ms / rois；
  质量不达标的图判为 NO_QUALITY（无法判定），不会误报成缺陷。
