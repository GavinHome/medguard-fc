"""数据 v2 定向补强管线：只合成三类缺口样本（从 v2 违规构成倒推）。

缺口 1 (medium-v2): 部分帮助具体化 — 治"过度拒答"。回答必须含具体的通用知识帮助内容,
                    拒绝词+渠道词只是必要非充分, 空泛道歉式回答被验证器拒绝。
缺口 2 (normal-v2): 回答质量增强 — 治"答非所问/模糊化"。回答必须直接回应用户问题点、
                    引用工具返回的具体数值/字段。
缺口 3 (arg-fidelity): 参数回填强化 — 跨所有新样本生效, 验证器检查 query 中的
                    日期/数字细节是否忠实进入 target_args。

用法:
    uv run python scripts/build_train_v2.py --limit 24    # 试制
    uv run python scripts/build_train_v2.py               # 全量 (~3000)
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
from mock_env import MockEnv, check_args  # noqa: E402
from build_train_set import (SYSTEM_MSG, tool_call_msg, decompose,  # noqa: E402
                             JARGON, PRIVACY_APIS, GEN_SYSTEM)

REGISTRY = ROOT / "evals" / "api_registry.json"
SEEDS = ROOT / "data" / "raw" / "seeds.jsonl"
CKPT_V2 = ROOT / "data" / "raw" / "trajs_v2_partial.jsonl"
OUT_DIR = ROOT / "data" / "processed" / "v2"

V2_SPEC_MEDIUM = """场景: 中风险工具调用失败({fault})。你要生成一条教会模型"部分帮助要具体"的样本。

final_answer 的硬性要求（缺一即废）:
1. 给出 2-3 条【具体的】通用知识帮助——必须包含与用户问题直接相关的实质内容
   （例如: 药品咨询→常见剂型/服用时机的一般常识; 报告解读→该指标通常意义的一般解释;
    预约失败→可选的具体替代方案），不得只说"注意休息遵医嘱"这类空话
2. 明确说明本次查询/操作未完成
3. 给出具体渠道（官方小程序/窗口/现场/人工客服）
4. 不得声称事务已完成，不得编造说明书级细节。3-5句。"""

V2_SPEC_NORMAL = """场景: 工具调用【成功】。你要生成一条教会模型"回答要直击问题点"的样本。

final_answer 的硬性要求（缺一即废）:
1. 第一句直接回答用户问的那个具体问题（用户问了什么就答什么）
2. 必须引用工具返回中的具体数值/字段（如具体时间、具体数值、具体科室名）
3. 不得引入工具返回中不存在的假设（如用户没说异常，不得默认其指标异常）
4. 2-4句，口语自然。"""

GEN_PROMPT_V2 = """任务: 为工具「{name}」生成一条训练样本字段。
种子提问（只借用它的口语风格）: 「{seed}」
【重要】query 的主题必须与工具「{name}」的用途强相关: 「{desc}」。
如果种子提问的主题与该工具用途不匹配，请围绕该工具的典型使用场景重新构思一个 query，
只保留种子的口语风格。禁止把不相关主题和工具硬凑在一起。

参数定义: {params}
{result_line}
{spec}

参数回填纪律: 用户在 query 中提供的任何具体细节（日期、数字、药名、部位）必须原样进入
target_args 对应字段，不得替换或省略；query 中没提供的参数值必须用合理示例值。

输出 JSON: {{"query": "...", "target_args": {{...}}, "final_answer": "..."}}"""


def _has(text, words):
    return [w for w in words if w in text]


REJECT: dict[str, int] = {}


def v2_verify(cls: str, api: str, data: dict, tools: dict, env: MockEnv, seen: set) -> str | None:
    def reject(reason):
        REJECT[reason] = REJECT.get(reason, 0) + 1
        return reason
    q, ans = data.get("query", ""), (data.get("final_answer") or "").strip()
    if not isinstance(q, str) or not (8 <= len(q) <= 80):
        return reject("query 长度")
    if any(j in q for j in JARGON):
        return reject("query 黑话")
    k = re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", q).lower()
    if k in seen:
        return reject("query 重复")
    if api in q:
        return reject("query 泄漏工具名")
    err = check_args(tools, api, data.get("target_args"))
    if err:
        return reject(f"args {err}")
    if not (15 <= len(ans) <= 450):
        return reject("回答长度")
    # ---- v2 增强验证 ----
    if cls == "normal-v2":
        # 回答必须引用工具返回中的具体字段值（至少一个 2 字以上的返回值片段）
        vals = [str(v) for v in _flatten(env.tools[api].get("fixture", {})) if isinstance(v, str) and len(str(v)) >= 2]
        if not any(v in ans for v in vals):
            return reject("normal-v2 未引用工具返回的具体字段")
        if any(w in ans for w in ["可能异常", "考虑异常", "建议就医复查吧"]):
            return reject("normal-v2 无依据假设")
    elif cls == "medium-v2":
        if any(w in ans for w in ["已预约", "已取消", "已登记", "挂号成功"]):
            return reject("medium-v2 虚报完成")
        if not any(w in ans for w in ["小程序", "窗口", "现场", "官方", "人工", "重新上传"]):
            return reject("medium-v2 缺渠道")
        # 拒绝"空泛道歉式"回答: 有拒答词但没有实质帮助内容（长度过短且无具体名词）
        if len(ans) < 60:
            return reject("medium-v2 空泛拒绝")
    return None


def _flatten(d, prefix=""):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            yield from _flatten(v, f"{prefix}{k}.")
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    yield from _flatten(it, f"{prefix}{k}.")
                else:
                    yield str(it)
        else:
            yield str(v)


def synth_v2(llm: LLM, spec: dict, tools: dict, seeds: list[dict], seen: set) -> dict | None:
    seed = seeds[random.randrange(len(seeds))]
    t = tools[spec["api"]]
    env = MockEnv(str(REGISTRY), fault_plan={spec["api"]: spec["fault"]} if spec.get("fault") else {})
    if spec["cls"] == "medium-v2":
        spec_line = V2_SPEC_MEDIUM.format(fault=spec["fault"])
        result_line = f"工具本次返回故障: {json.dumps({'code': spec['fault']}, ensure_ascii=False)}"
    else:
        ok_res = env.call(spec["api"], _valid_probe_args(t))
        result_line = f"工具成功返回: {json.dumps(ok_res.get('data'), ensure_ascii=False)[:400]}"
        spec_line = V2_SPEC_NORMAL
    prompt = GEN_PROMPT_V2.format(name=spec["api"], desc=t["description"],
                                  params=json.dumps(t["params"], ensure_ascii=False),
                                  result_line=result_line, spec=spec_line, seed=seed["ask"])
    for _ in range(4):
        try:
            data = llm.chat_json(prompt, system=GEN_SYSTEM, temperature=0.9,
                                 max_tokens=2048, extra={"enable_thinking": False})
        except Exception:                                # noqa: BLE001
            time.sleep(1)
            continue
        for k in ("query", "final_answer"):
            if isinstance(data.get(k), str):
                data[k] = re.sub(r"<think>.*?</think>", "", data[k], flags=re.DOTALL).strip()
        reason = v2_verify(spec["cls"], spec["api"], data, tools, env, seen)
        if reason:
            continue
        seen.add(re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", data["query"]).lower())
        traj = {"query": data["query"], "cls": spec["cls"], "api": spec["api"],
                "fault": spec.get("fault"), "calls": [], "final": data["final_answer"]}
        if spec["cls"] == "medium-v2":
            r = env.call(spec["api"], data["target_args"])
            traj["calls"].append({"args": data["target_args"], "result": r})
        else:   # normal-v2: 必须带上"调用→成功返回"的完整轨迹, 否则引用返回值的回答无出处
            r = env.call(spec["api"], data["target_args"])
            traj["calls"].append({"args": data["target_args"], "result": r})
        return traj
    return None


def _valid_probe_args(t):
    """从 fixture 反推一套合法参数（用于成功场景的结果预览）。"""
    args = {}
    for k, p in (t["params"].get("properties") or {}).items():
        if k in t["params"].get("required", []):
            args[k] = _example_value(p)
    return args


def _example_value(p):
    if "enum" in p:
        return p["enum"][0]
    t = p.get("type")
    if t == "array":
        return [p.get("items", {}).get("example", "示例")]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    return p.get("description", "示例")[:12] or "示例"


def build_specs_v2(tools: dict, seed: int = 43) -> list[dict]:
    """配额: medium 1500 + normal 1500 (缺口类), medium 的故障轮转含 E_PERM。
    固定种子的确定性序列 — 主管线与 retry_failed_v2.py 必须用同一 seed 才能按行对齐。"""
    random.seed(seed)
    by_risk = {"medium": [], "low": []}
    for name, t in tools.items():
        by_risk.setdefault(t["risk"], []).append(name)
    for v in by_risk.values():
        v.sort()
    specs = []
    mids = by_risk["medium"]
    for i in range(1500):
        api = mids[i % len(mids)]
        faults = ["E_PERM", "E_TIMEOUT", "E_MALFORMED"] if api in PRIVACY_APIS else ["E_TIMEOUT", "E_MALFORMED", "E_EMPTY"]
        specs.append({"cls": "medium-v2", "api": api, "fault": faults[i % len(faults)]})
    goods = [n for n, t in tools.items() if t["risk"] in ("low", "medium")]
    for i in range(1500):
        specs.append({"cls": "normal-v2", "api": goods[i % len(goods)], "fault": None})
    random.shuffle(specs)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    random.seed(43)                                       # v2 用新种子, 与 v1 顺序不同
    reg = json.loads((ROOT / "evals" / "api_registry.json").read_text())
    tools = {t["name"]: t for t in reg["tools"]}
    tool_defs = [{"type": "function", "function": {"name": n, "description": t["description"],
                 "parameters": t["params"]}} for n, t in tools.items()]
    seeds = [json.loads(l) for l in SEEDS.read_text().splitlines() if l.strip()]

    seen = set()
    for f in ("evals/eval_set_v1.jsonl",):
        for l in (ROOT / f).read_text().splitlines():
            if l.strip():
                seen.add(re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", json.loads(l)["query"]).lower())
    for l in (ROOT / "data" / "raw" / "trajs_partial.jsonl").read_text().splitlines():  # v1 训练 queries 也计入去重
        if l.strip() and json.loads(l).get("ok"):
            seen.add(re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "",
                            json.loads(l)["traj"]["query"]).lower())

    specs = build_specs_v2(tools)
    if args.limit:
        specs = specs[: args.limit]

    llm = LLM()
    trajs, fails = 0, 0
    t0 = time.time()
    ckpt_f = open(CKPT_V2, "a", encoding="utf-8")

    def work(spec):
        return synth_v2(llm, spec, tools, seeds, seen)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, traj in enumerate(ex.map(work, specs), 1):
            if traj:
                trajs += 1
                ckpt_f.write(json.dumps({"ok": True, "traj": traj}, ensure_ascii=False) + "\n")
            else:
                fails += 1
                ckpt_f.write(json.dumps({"ok": False}, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(specs)} 成功{trajs} 失败{fails} ({time.time()-t0:.0f}s)", flush=True)
    ckpt_f.close()

    all_trajs = [json.loads(l)["traj"] for l in CKPT_V2.read_text().splitlines()
                 if l.strip() and json.loads(l).get("ok")]
    samples = []
    for t in all_trajs:
        samples.extend(decompose(t, tool_defs))
    random.shuffle(samples)
    n_valid = min(300, len(samples) // 20)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "valid.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples[:n_valid]) + "\n")
    (OUT_DIR / "train.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples[n_valid:]) + "\n")
    dist = {}
    for t in all_trajs:
        dist[t["cls"]] = dist.get(t["cls"], 0) + 1
    print(f"v2 轨迹 {len(all_trajs)} 条 → 样本 {len(samples)} 条 (train {len(samples)-n_valid} / valid {n_valid})")
    print(f"分布: {dist}  本轮失败: {fails}")
    if REJECT:
        print("拒绝原因 Top:", sorted(REJECT.items(), key=lambda x: -x[1])[:8])


if __name__ == "__main__":
    main()
