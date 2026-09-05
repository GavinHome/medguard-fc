# MedGuard-FC — AutoDL 4090 训练操作清单（8B × v12 全量 → Gate 终对比）

> 目标: 在 4090 上完成 8B × v12 全量 (32,151 样本, 1 epoch) 训练,
> 产物下载回 Mac 做量化 + 冻结评测集终评 → Gate 判定 (B<5% / A2≥85% / A1≥74.7)。
> 预计: 训练 6-10h + 下载 1h + Mac 量化评测 1h · 成本约 ¥15-30。

## 1. 开机

- AutoDL 租用: RTX 4090 (24GB) × 1, 地区任意(选便宜的)
- 镜像: **PyTorch 2.5.1 + Python 3.11 + CUDA 12.4**（或更高的官方镜像）
- 数据盘 ≥ 40GB（基座 16GB + 输出 17GB + 数据 1GB）

## 2. 上传

把本目录整个上传（AutoDL 网盘中转或 scp 均可），目标位置 `/root/autodl-tmp/medguard/`:

```
medguard/
├── train_4090.py
├── requirements.txt
└── data/v12full/{train.jsonl, valid.jsonl}
```

## 3. 环境（容器内执行）

```bash
cd /root/autodl-tmp/medguard
pip install -r requirements.txt
# 基座走镜像下载 (16GB, 数据盘)
export HF_ENDPOINT=https://hf-mirror.com
```

## 4. 训练

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train_4090.py --data_dir data/v12full --out out/8b_v12 2>&1 | tee train.log
```

- 正常信号: 开头打印 `train 32151 / valid 400` 和 `trainable params ≈ 100M`
- 每 50 步打一次 loss（应从 ~2.5 降到 0.8-1.1 区间）
- 每 1000 步自动存档; 若中断: 直接重跑同命令会从最近 checkpoint 续（Trainer resume_from_checkpoint 需手动:
  在 Trainer(...) 调用里加 `resume_from_checkpoint=True` 或重跑时改用
  `python train_4090.py ... ` 后观察——为简单起见, 中断就重跑, 6-10h 在单次租期内通常够）
- 预计 6-10 小时

## 5. 合并 LoRA 并导出（容器内）

```bash
python - <<'EOF'
import torch, os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, "out/8b_v12/adapter")
m = m.merge_and_unload()
m.save_pretrained("out/8b_v12/merged_bf16")
AutoTokenizer.from_pretrained(base).save_pretrained("out/8b_v12/merged_bf16")
print("merged saved:", os.listdir("out/8b_v12/merged_bf16"))
EOF
```

## 6. 下载回 Mac

只需下载 `out/8b_v12/adapter/`（约 600MB，备查）和 `out/8b_v12/merged_bf16/`（约 16GB）。

AutoDL 下载方式（任选）:
- 网盘中转: 容器内把 merged_bf16 打包 `tar czf merged.tgz out/8b_v12/merged_bf16` 后上传网盘, Mac 端下载
- scp: `scp -P <端口> -r root@<主机>.autodl.com:/root/autodl-tmp/medguard/out/8b_v12/merged_bf16 .`

放到 Mac 的: `/Users/yangxiaomin/ZCodeProject/medguard-fc/models/qwen3-8b-v12-full-bf16/`

## 7. Mac 端量化 + 终评（我来执行）

```bash
uv run mlx_lm.convert --hf-path models/qwen3-8b-v12-full-bf16 -q --quantize-preset mixed_4bit \
  --mlx-path models/qwen3-8b-v12-full-4bit
uv run python -u scripts/run_eval.py --model models/qwen3-8b-v12-full-4bit --eval-set evals/eval_set_v1.jsonl
# + 安全提示词配置 + v2 裁判复核 → Gate 判定表
```

## 注意事项

- 训练前 `nvidia-smi` 确认 24GB 空闲; OOM 就 `--batch 1 --accum 8`
- 帐单按小时, 训完尽快合并+打包+关机（合并只需 CPU, 可在关机前完成）
- 不要升级/降级镜像里的 torch
