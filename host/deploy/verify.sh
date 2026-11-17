#!/usr/bin/env bash
# 板端自检：./verify.sh [安装目录]
# 检查项：二进制可执行、样例检测完成、判定结果合理、打印耗时。
set -euo pipefail

PREFIX="${1:-/opt/glue}"
cd "$PREFIX"

if [ ! -x bin/glue-agent ]; then
  echo "[FAIL] 找不到可执行文件 $PREFIX/bin/glue-agent"
  exit 1
fi

echo "== 1. 架构与二进制 =="
echo "板端架构：$(uname -m)"
file bin/glue-agent 2>/dev/null || true

echo "== 2. 运行样例检测（$(ls samples/*.png 2>/dev/null | wc -l) 张）=="
START=$(date +%s%3N 2>/dev/null || date +%s)
./bin/glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 --out logs/verify.json >/dev/null
END=$(date +%s%3N 2>/dev/null || date +%s)
echo "输出：$PREFIX/logs/verify.json"

echo "== 3. 判定统计 =="
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import collections, json, sys
d = json.load(open("logs/verify.json"))
c = collections.Counter(r["verdict"] for r in d["results"])
print("总数:", d["count"], " 判定:", dict(c))
expected = {"OK": 1, "TOO_NARROW": 1, "TOO_WIDE": 1, "BROKEN": 1,
            "NO_ROI": 1, "NO_QUALITY": 3}
bad = {k: (expected.get(k, 0), c.get(k, 0)) for k in set(expected) | set(c)
       if expected.get(k, 0) != c.get(k, 0)}
if bad:
    print("[FAIL] 判定分布与样例预期不符（expected, actual）:", bad)
    sys.exit(1)
print("[PASS] 8 张样例判定分布符合预期（1 正常 / 1 过窄 / 1 过宽 / 1 断胶 / 1 无胶 / 3 无法判定）")
PY
else
  echo "(未安装 python3，跳过判定统计；可直接查看 logs/verify.json)"
fi

echo "== 4. 结果 =="
if [ -s logs/verify.json ]; then
  echo "[PASS] 样例检测完成，耗时 ${START}→${END}"
  exit 0
else
  echo "[FAIL] 未生成 logs/verify.json"
  exit 1
fi
