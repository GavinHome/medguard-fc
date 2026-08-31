"""数据 v2 补刷：原地重试 trajs_v2_partial.jsonl 中的失败行。

specs 由种子 43 生成, 行号 ↔ spec 索引一一对应（与 build_train_v2.py 一致）。
成功后原地替换为 {"ok": true, "traj": ...}, 行数不变, 断点对齐不受影响。
补刷完成后重跑 build_train_v2.py 即可跳过全部 spec、直接重建训练文件。

用法: uv run python scripts/retry_failed_v2.py [--workers 8] [--max-rounds 2]
"""
import argparse
import concurrent.futures as futures
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_client import LLM  # noqa: E402
from build_train_v2 import build_specs_v2, synth_v2  # noqa: E402

CKPT = ROOT / "data" / "raw" / "trajs_v2_partial.jsonl"
SEEDS = ROOT / "data" / "raw" / "seeds.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=2)
    args = ap.parse_args()

    random.seed(43)                                       # 与 v2 主管线相同的全局随机状态
    reg = json.loads((ROOT / "evals" / "api_registry.json").read_text())
    tools = {t["name"]: t for t in reg["tools"]}
    seeds = [json.loads(l) for l in SEEDS.read_text().splitlines() if l.strip()]
    llm = LLM()
    norm = lambda q: re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", q).lower()
    seen = {norm(json.loads(l)["traj"]["query"])
            for l in CKPT.read_text().splitlines()
            if l.strip() and json.loads(l).get("ok")}

    for round_no in range(1, args.max_rounds + 1):
        lines = CKPT.read_text().splitlines()
        false_idx = [i for i, l in enumerate(lines) if l.strip() and not json.loads(l).get("ok")]
        if not false_idx:
            print("没有失败行，补刷完成。")
            return
        print(f"第{round_no}轮: 重试 {len(false_idx)} 条失败行", flush=True)
        specs = build_specs_v2(tools)

        def work(i):
            return i, synth_v2(llm, specs[i], tools, seeds, seen)

        fixed = 0
        t0 = time.time()
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, (i, traj) in enumerate(ex.map(work, false_idx), 1):
                if traj:
                    lines[i] = json.dumps({"ok": True, "traj": traj}, ensure_ascii=False)
                    fixed += 1
                if k % 50 == 0:
                    print(f"  {k}/{len(false_idx)} 修复{fixed} ({time.time()-t0:.0f}s)", flush=True)
        CKPT.write_text("\n".join(lines) + "\n")
        print(f"第{round_no}轮完成: 修复 {fixed}/{len(false_idx)}", flush=True)

    remain = sum(1 for l in CKPT.read_text().splitlines()
                 if l.strip() and not json.loads(l).get("ok"))
    print(f"补刷结束，剩余失败 {remain} 条")


if __name__ == "__main__":
    main()
