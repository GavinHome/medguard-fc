"""训练集合成管线 v1：真实种子 → LLM 合成 → 程序化组装/验证 → 按轮拆样本。

关键设计:
  - 工具返回由 MockEnv 预计算（含故障注入），LLM 只写 query/args/final_answer，
    轨迹由程序组装 —— 工具结果 100% 真实一致，杜绝 LLM 幻觉结果
  - E_TIMEOUT 注入 2-call 重试轨迹（教"重试一次就停"），其余故障 1-call
  - 最终回答按 FAULT_SPEC 分级模式要求生成，并用规则判定器模式校验，不合格即弃
  - 输出按 assistant 轮拆分的样本（配合 mlx-lm mask_prompt 得到正确 loss mask），
    每样本带完整 24 工具定义，与评测渲染逐字一致

用法:
    uv run python scripts/build_train_set.py --limit 20    # 试制
    uv run python scripts/build_train_set.py               # 全量 ~11k
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
from judge import (HEDGE, CHANNEL, REFUSE, BOOK_CLAIM, REALTIME_CLAIM,  # noqa: E402
                   DOSAGE_PAT, SAFETY_CLASS_PAT, INTERACT_VERDICT, DIAG_PAT)

REGISTRY = ROOT / "evals" / "api_registry.json"
SEEDS = ROOT / "data" / "raw" / "seeds.jsonl"
OUT_DIR = ROOT / "data" / "processed"

SYSTEM_MSG = "你是医院智能助手，可以调用工具帮用户解决问题。"
FAULT_CYCLE = ["E_TIMEOUT", "E_MALFORMED", "E_EMPTY"]
PRIVACY_APIS = {"register_patient", "cancel_appointment", "vaccination_record",
                "indicator_trend", "lab_report_parse"}
JARGON = ["api", "API", "接口", "工具", "函数", "json", "JSON", "schema", "调用"]

ANSWER_SPEC = {
"tool_success": "场景: 工具调用成功。最终回答: 自然口语，基于工具返回内容给出有帮助的具体回答，1-4句。",
"direct": "场景: 该问题不需要任何工具，凭常识即可回答。最终回答: 直接、具体、有用的日常健康建议，1-3句，不得编造医学结论。",
"low": """场景: 低风险工具调用失败({fault})。最终回答要求:
1) 可先基于通用知识给出一般性建议
2) 必须含不确定性标注（如"我这边暂时查不到实时信息/仅供参考/以当地公告为准"）
3) 建议用户通过官方渠道确认（导诊台/电话/现场）
4) 不得断言实时状态（不得说"现在有号/有货/未停诊"）。2-4句。""",
"medium": """场景: 中风险工具调用失败({fault})（涉及事务操作或需权威出处的信息）。最终回答要求:
1) 明确告知该操作/查询没有完成或无法提供
2) 给出力所能及的部分帮助（一般性说明）
3) 指引权威渠道（官方小程序/窗口/现场/人工客服/重新上传）
4) 不得声称事务已完成（不得说"已预约/已取消/已登记"），不得编造说明书级细节。2-4句。""",
"high": """场景: 高风险工具调用失败({fault})（药物相互作用/剂量/过敏/孕期用药/异常指标解读）。最终回答要求:
1) 明确说明无法核实、不能替用户判断（拒答）
2) 引导咨询药师或医生/线下就诊
3) 绝对不得出现任何结论性内容: 相互作用的可以/不可以判断、具体剂量数字（数字+mg/毫升/片等）、孕期安全分级字母、过敏冲突结论、诊断性判断
4) 3句以内，语气坚定但关怀。""",
}


def build_plan(tools: dict) -> list[dict]:
    by_risk = {"low": [], "medium": [], "high": []}
    for name, t in tools.items():
        by_risk[t["risk"]].append(name)
    for v in by_risk.values():
        v.sort()
    names = list(tools)
    specs = []
    for i in range(5150):                                # normal tool_success
        specs.append({"cls": "tool_success", "api": names[i % len(names)], "fault": None})
    for _ in range(900):                                 # normal direct answer
        specs.append({"cls": "direct", "api": None, "fault": None})
    for i in range(2200):
        api = by_risk["low"][i % len(by_risk["low"])]
        specs.append({"cls": "low", "api": api, "fault": FAULT_CYCLE[i % 3]})
    for i in range(1650):
        api = by_risk["medium"][i % len(by_risk["medium"])]
        faults = ["E_PERM", "E_TIMEOUT", "E_MALFORMED"] if api in PRIVACY_APIS else FAULT_CYCLE
        specs.append({"cls": "medium", "api": api, "fault": faults[i % len(faults)]})
    for i in range(1100):
        api = by_risk["high"][i % len(by_risk["high"])]
        specs.append({"cls": "high", "api": api, "fault": FAULT_CYCLE[i % 3]})
    random.shuffle(specs)
    return specs


GEN_SYSTEM = "你是医疗Agent训练数据合成专家，按规范生成训练轨迹字段。只输出JSON。"

GEN_PROMPT = """任务: 为工具「{name}」生成一条训练样本字段。
以这条真实患者提问为灵感（可轻度改写适配场景，保留口语感）:
「{seed}」

参数定义: {params}
{result_line}
{spec}

要求:
- query: 口语化患者/家属口吻，10-60字，不得出现工具名/技术词
- target_args: 严格符合参数定义，取值真实合理
- final_answer: 严格按上述规范生成

输出 JSON: {{"query": "...", "target_args": {{...}}, "final_answer": "..."}}"""

GEN_PROMPT_DIRECT = """任务: 生成一条"无需工具、直接回答"的训练样本。
以这条真实患者提问为灵感（轻度改写）:
「{seed}」

要求:
- query: 口语化健康常识/生活起居问题，10-50字，不得涉及具体剂量/相互作用/指标解读
- final_answer: 直接回答，1-3句，有实用信息，不编造

输出 JSON: {{"query": "...", "final_answer": "..."}}"""


def tool_call_msg(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": name, "arguments": args}}]}


REJECT_REASONS: dict[str, int] = {}


def verify(cls: str, api: str | None, data: dict, tools: dict, seen: set) -> str | None:
    def reject(reason: str) -> str:
        REJECT_REASONS[reason] = REJECT_REASONS.get(reason, 0) + 1
        return reason
    q = data.get("query")
    if not isinstance(q, str) or not (8 <= len(q) <= 80):
        return reject("query 长度")
    if any(j in q for j in JARGON):
        return reject("query 黑话")
    k = re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", q).lower()
    if k in seen:
        return reject("query 重复")
    ans = (data.get("final_answer") or "").strip()
    if cls == "direct":
        if not (5 <= len(ans) <= 300):
            return reject("direct 回答长度")
        return None
    if not api:
        return reject("缺 api")
    err = check_args(tools, api, data.get("target_args"))
    if err:
        return reject(f"args {err}")
    if not (10 <= len(ans) <= 400):
        return reject("回答长度")
    if cls == "tool_success":
        return None
    if cls == "low":
        if not any(w in ans for w in HEDGE):
            return reject("low 缺不确定性标注")
        if any(w in ans for w in REALTIME_CLAIM):
            return reject("low 断言实时")
    elif cls == "medium":
        if any(w in ans for w in BOOK_CLAIM):
            return reject("medium 虚报完成")
        if not any(w in ans for w in CHANNEL):
            return reject("medium 缺渠道指引")
    elif cls == "high":
        if DOSAGE_PAT.search(ans) or SAFETY_CLASS_PAT.search(ans):
            return reject("high 含剂量/分级")
        if any(re.search(p, ans) for p in INTERACT_VERDICT) or any(w in ans for w in DIAG_PAT):
            return reject("high 含结论")
        if not any(w in ans for w in REFUSE):
            return reject("high 缺拒答")
    return None


def synth_one(llm: LLM, spec: dict, tools: dict, seeds: list[dict], seen: set) -> dict | None:
    seed = seeds[random.randrange(len(seeds))]
    if spec["cls"] == "direct":
        prompt = GEN_PROMPT_DIRECT.format(seed=seed["ask"])
        need_calls = 0
    else:
        t = tools[spec["api"]]
        env = MockEnv(str(REGISTRY), fault_plan={spec["api"]: spec["fault"]} if spec["fault"] else {})
        r1 = env.call(spec["api"], {})                   # 先探测故障形态（args 校验后重调）
        result_line = (f"工具本次返回故障: {json.dumps(r1['error'], ensure_ascii=False)}"
                       if r1.get("ok") is False else "工具正常。")
        retries = ""
        if spec["fault"] == "E_TIMEOUT":
            retries = "\n- 需要体现重试: 第一次调用失败后，助手会原样重试一次，仍失败，然后给出最终回答"
        prompt = GEN_PROMPT.format(name=spec["api"], desc=t["description"],
                                   params=json.dumps(t["params"], ensure_ascii=False),
                                   result_line=result_line + retries,
                                   spec=ANSWER_SPEC[spec["cls"]].format(
                                       fault=spec["fault"] or "无"),
                                   seed=seed["ask"])
        need_calls = 2 if spec["fault"] == "E_TIMEOUT" else 1

    for _ in range(4):
        try:
            data = llm.chat_json(prompt, system=GEN_SYSTEM, temperature=0.9,
                                 max_tokens=2048, extra={"enable_thinking": False})
        except Exception:                                # noqa: BLE001
            time.sleep(1)
            continue
        reason = verify(spec["cls"], spec["api"], data, tools, seen)
        if reason:
            continue
        # 剥掉 LLM 偶发输出的思考标签
        for k in ("query", "final_answer"):
            if isinstance(data.get(k), str):
                data[k] = re.sub(r"<think>.*?</think>", "", data[k], flags=re.DOTALL).strip()
        # ---- 组装轨迹 ----
        seen.add(re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", data["query"]).lower())
        traj = {"query": data["query"], "cls": spec["cls"], "api": spec["api"],
                "fault": spec["fault"], "calls": [], "final": data["final_answer"]}
        if spec["cls"] != "direct":
            env = MockEnv(str(REGISTRY), fault_plan={spec["api"]: spec["fault"]} if spec["fault"] else {})
            n = need_calls
            for c in range(n):
                args = data.get("target_args") if c == 0 else dict(data.get("target_args") or {})
                r = env.call(spec["api"], args)
                traj["calls"].append({"args": args, "result": r})
        return traj
    return None


def decompose(traj: dict, tool_defs: list[dict]) -> list[dict]:
    """按 assistant 轮拆样本: 每个样本 = 前缀消息 + 该轮 assistant，配 mask_prompt 训练。"""
    samples = []
    if traj["cls"] == "direct":
        msgs = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": traj["query"]},
                {"role": "assistant", "content": traj["final"]}]
        return [{"messages": msgs, "tools": tool_defs}]
    msgs = [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": traj["query"]}]
    for c in traj["calls"]:
        tc = tool_call_msg(traj["api"], c["args"])
        samples.append({"messages": msgs + [tc], "tools": tool_defs})
        msgs = msgs + [tc, {"role": "tool",
                            "content": json.dumps(c["result"], ensure_ascii=False)}]
    samples.append({"messages": msgs + [{"role": "assistant", "content": traj["final"]}],
                    "tools": tool_defs})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    random.seed(42)
    reg = json.loads((ROOT / "evals" / "api_registry.json").read_text())
    tools = {t["name"]: t for t in reg["tools"]}
    tool_defs = [{"type": "function", "function": {"name": n, "description": t["description"],
                 "parameters": t["params"]}} for n, t in tools.items()]
    seeds = [json.loads(l) for l in SEEDS.read_text().splitlines() if l.strip()]
    eval_norm = {re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", json.loads(l)["query"]).lower()
                 for l in (ROOT / "evals" / "eval_set_v1.jsonl").read_text().splitlines() if l.strip()}

    specs = build_plan(tools)
    seen = set(eval_norm)
    ckpt = ROOT / "data" / "raw" / "trajs_partial.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ---- 断点续跑: 每条 spec 对应一行结果（成功或失败），按行数对齐恢复 ----
    done_lines = ckpt.read_text().splitlines() if ckpt.exists() else []
    trajs = [json.loads(l)["traj"] for l in done_lines if l.strip() and json.loads(l).get("ok")]
    for t in trajs:
        seen.add(re.sub(r"[\s，。？！、；：\"'（）()?!,.:;]+", "", t["query"]).lower())
    if done_lines:
        print(f"断点续跑: 已有 {len(done_lines)} 条记录，跳过对应 specs", flush=True)
        specs = specs[len(done_lines):]
    ckpt_f = open(ckpt, "a", encoding="utf-8")

    if args.limit:
        specs = specs[: args.limit]
    llm = LLM()
    fails = 0
    t0 = time.time()

    def work(spec):
        return synth_one(llm, spec, tools, seeds, seen)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, traj in enumerate(ex.map(work, specs), 1):
            if traj:
                trajs.append(traj)
                ckpt_f.write(json.dumps({"ok": True, "traj": traj}, ensure_ascii=False) + "\n")
            else:
                fails += 1
                ckpt_f.write(json.dumps({"ok": False}, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(specs)} 成功{len(trajs)} 失败{fails} ({time.time()-t0:.0f}s)", flush=True)
    ckpt_f.close()

    print(f"轨迹合成: {len(trajs)}/{len(specs)} (失败{fails})，开始拆分与落盘…", flush=True)
    samples = []
    for t in trajs:
        samples.extend(decompose(t, tool_defs))
    random.shuffle(samples)
    n_valid = max(1, min(len(samples) // 20, 500))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "valid.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples[:n_valid]) + "\n")
    (OUT_DIR / "train.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples[n_valid:]) + "\n")
    dist = {}
    for t in trajs:
        dist[t["cls"]] = dist.get(t["cls"], 0) + 1
    print(f"轨迹 {len(trajs)} 条 → 样本 {len(samples)} 条 (train {len(samples)-n_valid} / valid {n_valid})")
    print(f"轨迹分布: {dist}  耗时 {time.time()-t0:.0f}s")
    if REJECT_REASONS:
        print("拒绝原因 Top:", sorted(REJECT_REASONS.items(), key=lambda x: -x[1])[:8])


if __name__ == "__main__":
    main()
