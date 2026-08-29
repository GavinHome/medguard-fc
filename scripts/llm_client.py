"""StepFun LLM 客户端（OpenAI 兼容协议，读 .env 配置）。

支持多账号负载均衡: 每个账号独立 RPM 节流窗口，轮询调度，429 自动换号重试。

.env 配置（两种写法兼容）:
  单账号（旧格式）:
    STEPFUN_API_KEY=...
    STEPFUN_BASE_URL=https://api.stepfun.com/v1
    STEPFUN_MODEL=step-3.7-flash
    STEPFUN_RPM=9
  多账号（新格式，编号从 1 递增）:
    STEPFUN_KEY_1=...
    STEPFUN_URL_1=https://api.stepfun.com/v1
    STEPFUN_RPM_1=9
    STEPFUN_KEY_2=...
    STEPFUN_URL_2=...
    STEPFUN_RPM_2=9

用法:
    from llm_client import LLM
    llm = LLM()
    text = llm.chat("...")
    data = llm.chat_json("输出 JSON: {...}")
"""
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import requests


def _load_env(env_path: Path | None = None) -> None:
    env_path = env_path or Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class _Account:
    """单个账号: 独立的 RPM 滑动窗口。"""

    def __init__(self, name: str, key: str, url: str, model: str, rpm: int):
        self.name, self.key, self.url, self.model = name, key, url.rstrip("/"), model
        self.rpm = int(rpm or 0)
        self.ts: deque = deque()
        self.lock = threading.Lock()

    def acquire(self):
        """尝试占用一个请求名额。返回 None=已占用名额立即可用；返回秒数=拒绝(不占名额)。"""
        with self.lock:
            now = time.time()
            while self.ts and now - self.ts[0] >= 60:
                self.ts.popleft()
            if self.rpm and len(self.ts) >= self.rpm:
                return max(0.05, 60 - (now - self.ts[0]) + 0.05)
            self.ts.append(time.time())
            return None


class LLM:
    def __init__(self):
        _load_env()
        self.accounts = self._load_accounts()
        if not self.accounts:
            raise RuntimeError(".env 中未找到 STEPFUN_API_KEY 或 STEPFUN_KEY_N 配置")
        self._rr = 0
        self._last_usage = None

    @staticmethod
    def _load_accounts() -> list:
        accs, i = [], 1
        while os.environ.get(f"STEPFUN_KEY_{i}"):
            accs.append(_Account(
                f"#{i}", os.environ[f"STEPFUN_KEY_{i}"],
                os.environ.get(f"STEPFUN_URL_{i}", "https://api.stepfun.com/v1"),
                os.environ.get(f"STEPFUN_MODEL_{i}", "step-3.7-flash"),
                os.environ.get(f"STEPFUN_RPM_{i}", "0")))
            i += 1
        if not accs and os.environ.get("STEPFUN_API_KEY"):
            accs.append(_Account(
                "#单账号", os.environ["STEPFUN_API_KEY"],
                os.environ.get("STEPFUN_BASE_URL", "https://api.stepfun.com/v1"),
                os.environ.get("STEPFUN_MODEL", "step-3.7-flash"),
                os.environ.get("STEPFUN_RPM", "0")))
        return accs

    def _acquire(self) -> _Account:
        """轮询各账号拿名额；全部在冷却时分段睡眠重探（探测不占名额）。"""
        while True:
            fallback = None
            for _ in range(len(self.accounts)):
                acc = self.accounts[self._rr % len(self.accounts)]
                self._rr += 1
                wait = acc.acquire()
                if wait is None:
                    return acc
                if fallback is None or wait < fallback[0]:
                    fallback = (wait, acc)
            time.sleep(min(fallback[0] if fallback else 1.0, 3.0))

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.8,
             max_tokens: int = 4096, retries: int = 4, extra: dict | None = None) -> str:
        """step-3.7-flash 是推理模型: 思考内容计入 max_tokens，配额太小会导致
        思考吃光额度、正文为空。默认 4096。extra: 附加请求字段。"""
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        last_err = None
        for attempt in range(retries):
            acc = self._acquire()
            payload = {"model": acc.model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}
            if extra:
                payload.update(extra)
            try:
                r = requests.post(f"{acc.url}/chat/completions", json=payload,
                                  headers={"Authorization": f"Bearer {acc.key}"},
                                  timeout=90)
                r.raise_for_status()
                data = r.json()
                self._last_usage = data.get("usage")
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:                       # noqa: BLE001
                last_err = e
                time.sleep(min(2 * (attempt + 1), 5))    # 429 时 _acquire 会自动换号
        raise RuntimeError(f"全部 {len(self.accounts)} 个账号调用失败（重试 {retries} 次/账号轮转）: {last_err}")

    def chat_json(self, prompt: str, **kw) -> dict | list:
        text = self.chat(prompt, **kw).strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(1))
