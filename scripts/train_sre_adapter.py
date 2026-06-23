#!/usr/bin/env python3
"""
Train the bundled SRE assistant LoRA adapter with the HF/CPU backend (so it's
compatible with how the appliance serves in-cluster). Produces a PEFT adapter at
models/adapters/sre-assistant from data/sre-tables-train/sre_qa.jsonl.

Usage:  python3 scripts/train_sre_adapter.py [base_model] [epochs]
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from pipeline.train_qlora import prepare_training_data

BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DATA = "data/sre-tables-train/sre_qa.jsonl"
RUN_DIR = "data/sre-assistant/run"
ADAPTER = "models/adapters/sre-assistant"

print(f"[train] preparing data from {DATA}")
prepare_training_data(verified_path=DATA, output_dir=RUN_DIR, test_split=0.1, max_examples=2000)

cfg = {
    "niche": "sre-assistant", "data_dir": RUN_DIR, "adapter_path": ADAPTER,
    "base_model": BASE, "lora_rank": 16, "lora_alpha": 32, "learning_rate": 1e-4,
    "batch_size": 4, "epochs": EPOCHS, "max_seq_length": 1024, "max_rows": 2000, "stop_file": "",
}
print(f"[train] base={BASE} epochs={EPOCHS} -> {ADAPTER}")
worker = os.path.join(ROOT, "pipeline", "training_worker_hf.py")
proc = subprocess.Popen([sys.executable, worker], stdin=subprocess.PIPE, text=True)
proc.stdin.write(json.dumps(cfg) + "\n"); proc.stdin.flush(); proc.stdin.close()
rc = proc.wait()
print(f"[train] worker exit {rc}")
print("[train] adapter files:", os.listdir(ADAPTER) if os.path.isdir(ADAPTER) else "MISSING")
