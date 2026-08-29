"""从老医疗项目 79.2 万条真实问答中挖掘合成种子（只用 ask 侧）。

过滤: 长度 12-80 字 / 去重 / 剔除与冻结评测集 query 重合的（防自我污染）
输出: data/raw/seeds.jsonl  {"id", "dept", "ask"}

用法: uv run python scripts/mine_seeds.py [--n 12000]
"""
import argparse
import csv
import glob
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = Path("/Volumes/SamsungSSD/Medical-SFT-Chatbot/中文医疗数据集")
EVAL_SET = ROOT / "evals" / "eval_set_v1.jsonl"


def norm(q: str) -> str:
    q = unicodedata.normalize("NFKC", q)
    return re.sub(r"[，。？！、；：\"'（）()\s?!,.:;]+", "", q).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    args = ap.parse_args()

    eval_norm = {norm(json.loads(l)["query"])
                 for l in EVAL_SET.read_text().splitlines() if l.strip()}

    files = glob.glob(str(OLD_DATA / "*" / "*.csv"))
    pool, seen = [], set()
    for f in files:
        with open(f, encoding="gb18030") as fh:
            for row in csv.DictReader(fh):
                ask = (row.get("ask") or "").strip()
                dept = (row.get("department") or "").strip()
                if not (12 <= len(ask) <= 80):
                    continue
                if re.search(r"http|www\.|\d{7,}", ask):     # 链接/长号码
                    continue
                k = norm(ask)
                if not k or k in seen or k in eval_norm:
                    continue
                seen.add(k)
                pool.append({"dept": dept, "ask": ask})
        print(f"{f.split('/')[-1]}: 累计候选 {len(pool)}", flush=True)

    random.seed(42)
    random.shuffle(pool)

    # 科室粗平衡采样（内科为主是数据本身分布，按科室轮转取）
    by_dept: dict[str, list] = {}
    for p in pool:
        by_dept.setdefault(p["dept"], []).append(p)
    depts = sorted(by_dept, key=lambda d: -len(by_dept[d]))
    picked, i = [], 0
    while len(picked) < args.n and any(by_dept[d] for d in depts):
        d = depts[i % len(depts)]
        if by_dept[d]:
            picked.append(by_dept[d].pop())
        i += 1

    out = ROOT / "data" / "raw" / "seeds.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps({"id": f"seed_{j:05d}", **p}, ensure_ascii=False)
                             for j, p in enumerate(picked)) + "\n")
    print(f"种子: {len(picked)} 条 → {out}  来源池 {len(pool)}")


if __name__ == "__main__":
    main()
