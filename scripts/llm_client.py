"""StepFun LLM 客户端（OpenAI 兼容协议，读 .env 配置）。

用于: 评测集合成、LLM-as-judge v2、阶段 2 训练数据合成。

用法:
    from llm_client import LLM
    llm = LLM()
    text = llm.chat("用一句话说明什么是感冒")
    data = llm.chat_json("输出 JSON: {\"a\": 1}")   # 自动剥 code fence + 解析
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


class LLM:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        _load_env()
        self.api_key = api_key or os.environ["STEPFUN_API_KEY"]
        self.base_url = (base_url or os.environ["STEPFUN_BASE_URL"]).rstrip("/")
        self.model = model or os.environ.get("STEPFUN_MODEL", "step-3.7-flash")
        self._last_usage = None
        # RPM 节流: 低档位 key（如 10 RPM）必须全进程共享节流，否则并发打爆窗口
        self.rpm = int(os.environ.get("STEPFUN_RPM", "0") or 0)
        self._ts: deque = deque()
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        if not self.rpm:
            return
        with self._lock:
            now = time.time()
            while self._ts and now - self._ts[0] >= 60:
                self._ts.popleft()
            wait = max(0.0, 60 - (now - self._ts[0]) + 0.05) if len(self._ts) >= self.rpm else 0.0
            self._ts.append(time.time())
        if wait > 0:
            time.sleep(wait)

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.8,
             max_tokens: int = 4096, retries: int = 3, extra: dict | None = None) -> str:
        """step-3.7-flash 是推理模型: 思考内容(reasoning)也计入 max_tokens，
        配额太小会导致思考吃光额度、正文为空(finish_reason=length)。默认 4096。
        extra: 附加请求字段（如 {"enable_thinking": False} 提速）。"""
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if extra:
            payload.update(extra)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err = None
        for attempt in range(retries):
            try:
                self._throttle()
                r = requests.post(f"{self.base_url}/chat/completions",
                                  json=payload, headers=headers, timeout=90)
                r.raise_for_status()
                data = r.json()
                self._last_usage = data.get("usage")
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:                       # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"StepFun 调用失败（重试 {retries} 次）: {last_err}")

    def chat_json(self, prompt: str, **kw) -> dict | list:
        text = self.chat(prompt, **kw).strip()
        # 剥 markdown code fence；再不行就截取首个 {...} / [...] 平衡块
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
