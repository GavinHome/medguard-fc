"""从清洗后的 v2 轨迹 (trajs_v2_clean.json) 重建训练文件。

用法: uv run python scripts/rebuild_v2.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_train_set import decompose  # noqa: E402

random.seed(44)
reg = json.loads((ROOT / "evals" / "api_registry.json").read_text())
tools = {t["name"]: t for t in reg["tools"]}
tool_defs = [{"type": "function", "function": {"name": n, "description": t["description"],
             "parameters": t["params"]}} for n, t in tools.items()]

trajs = json.load(open(ROOT / "data" / "raw" / "trajs_v2_clean.json"))
samples = []
for t in trajs:
    samples.extend(decompose(t, tool_defs))
random.shuffle(samples)
n_valid = min(300, len(samples) // 20)
out = ROOT / "data" / "processed" / "v2"
out.mkdir(parents=True, exist_ok=True)
(out / "valid.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples[:n_valid]) + "\n")
(out / "train.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples[n_valid:]) + "\n")
print(f"v2 轨迹 {len(trajs)} → 样本 {len(samples)} (train {len(samples)-n_valid} / valid {n_valid}) → {out}")
