"""MedGuard-FC 8B × v12 全量 — AutoDL 4090 QLoRA 训练脚本。

与 Mac mlx-lm 语义对齐: 同一 chat template(带 tools)、仅 assistant 轮次计 loss、
LoRA rank 32 / alpha 64 / lr 1e-4、无 packing。

用法: python train_4090.py --data_dir data/v12full --out out/8b_v12
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


def render(tokenizer, messages, tools, max_len):
    """逐轮次渲染 + 差分: assistant 轮的增量段(角色头+内容+im_end)计 loss。

    与 mlx-lm ChatDataset(mask_prompt)同语义: 模型只学"怎么答", 不学"怎么问"。
    """
    ids_per_turn = [
        tokenizer.apply_chat_template(messages[: i + 1], tools=tools,
                                      return_dict=False)
        for i in range(len(messages))
    ]
    full = ids_per_turn[-1]
    labels = [-100] * len(full)
    prev = 0
    for i, msg in enumerate(messages):
        cur = len(ids_per_turn[i])
        if msg["role"] == "assistant":
            for j in range(prev, cur):
                labels[j] = full[j]
        prev = cur
    return full[-max_len:], labels[-max_len:]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_len=2560):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, labels = render(self.tok, r["messages"], r.get("tools"), self.max_len)
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in feats:
            pad = n - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            out["labels"].append(f["labels"] + [-100] * pad)
            out["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/v12full")
    ap.add_argument("--out", default="out/8b_v12")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--max_len", type=int, default=2560)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=bnb)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = SFTDataset(os.path.join(args.data_dir, "train.jsonl"), tok, args.max_len)
    val = SFTDataset(os.path.join(args.data_dir, "valid.jsonl"), tok, args.max_len)
    print(f"train {len(ds)} / valid {len(val)}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=2000,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False)
    Trainer(model=model, args=targs, train_dataset=ds, eval_dataset=val,
            data_collator=Collator(tok.pad_token_id or tok.eos_token_id)).train()

    model.save_pretrained(os.path.join(args.out, "adapter"))
    print("训练完成, adapter 已保存至", os.path.join(args.out, "adapter"))


if __name__ == "__main__":
    main()
