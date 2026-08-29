"""模拟医疗 API 环境：注册表驱动的工具执行器 + 故障注入。

被 run_eval.py（评测）和阶段 2 的数据合成管线共用。

用法:
    env = MockEnv("evals/api_registry.json", fault_plan={"drug_interaction_check": "E_TIMEOUT"})
    env.tool_defs()          # 传给 chat template 的工具定义
    env.call("drug_interaction_check", {"drugs": ["A", "B"]})   # 返回 dict
    env.log                  # 全部调用记录（判定器依据）
"""
import copy
import json


class MockEnv:
    def __init__(self, registry_path: str, fault_plan: dict | None = None):
        with open(registry_path) as f:
            reg = json.load(f)
        self.tools = {t["name"]: t for t in reg["tools"]}
        self.fault_plan = dict(fault_plan or {})
        self.log: list[dict] = []

    def tool_defs(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": n, "description": t["description"], "parameters": t["params"]}}
            for n, t in self.tools.items()
        ]

    def call(self, name: str, arguments: dict | None) -> dict:
        arguments = arguments or {}
        entry = {"tool": name, "arguments": arguments, "ok": False, "error": None, "data": None}
        self.log.append(entry)

        t = self.tools.get(name)
        if t is None:
            entry["error"] = "E_NO_SUCH_TOOL"
            return _err("E_NO_SUCH_TOOL", f"未注册的工具: {name}")

        # 参数校验：E_PARAM 由模型自身触发，不可注入
        err = check_args(self.tools, name, arguments)
        if err:
            entry["error"] = err
            return _err(err, f"{name} 参数校验失败: {arguments}")

        # 故障注入（E_TIMEOUT / E_MALFORMED / E_EMPTY / E_PERM）
        if name in self.fault_plan:
            entry["error"] = self.fault_plan[name]
            return _err(self.fault_plan[name], f"{name} 模拟故障: {self.fault_plan[name]}")

        entry["ok"] = True
        entry["data"] = copy.deepcopy(t.get("fixture", {}))
        return {"ok": True, "data": entry["data"]}

    # ---- 查询辅助 ----
    def calls_to(self, name: str) -> list[dict]:
        return [e for e in self.log if e["tool"] == name]

    @property
    def num_calls(self) -> int:
        return len(self.log)


def check_args(tools: dict, name: str, arguments: dict | None) -> str | None:
    """独立参数校验（合成管线复用）。返回错误码或 None。"""
    t = tools.get(name)
    if t is None:
        return "E_NO_SUCH_TOOL"
    arguments = arguments or {}
    props = t["params"].get("properties", {})
    for req in t["params"].get("required", []):
        if req not in arguments or arguments[req] in (None, "", []):
            return "E_PARAM"
    for k, v in arguments.items():
        if k not in props:
            continue
        want = props[k].get("type")
        type_ok = {
            "string": isinstance(v, str),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "array": isinstance(v, list),
            "boolean": isinstance(v, bool),
        }.get(want, True)
        enum = props[k].get("enum")
        if enum and v not in enum:
            return "E_PARAM"
        if not type_ok:
            return "E_PARAM"
    return None


def _err(code: str, msg: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": msg}}
