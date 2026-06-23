#!/usr/bin/env python3
"""Wait for SRE adapter training, merge it (HF path), and generate to verify."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

ADAPTER = "models/adapters/sre-assistant"
# 1) wait for training to finish (adapter_config.json appears at save time)
print("[verify] waiting for adapter…", flush=True)
for _ in range(180):
    if os.path.exists(os.path.join(ADAPTER, "adapter_config.json")):
        break
    time.sleep(5)
else:
    print("[verify] TIMEOUT waiting for adapter"); sys.exit(1)
time.sleep(3)
print("[verify] adapter present:", sorted(os.listdir(ADAPTER)), flush=True)

# 2) merge via the HF exporter (the in-cluster path; bypasses MLX dispatch)
from pipeline.export_hf import export_model
merged = export_model("sre-assistant", ADAPTER, register=False)
print("[verify] merged dir:", merged, flush=True)
assert merged and os.path.isdir(merged), "merge failed"
print("[verify] merged files:", sorted(os.listdir(merged))[:12], flush=True)

# 3) load merged model and generate on SRE questions
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.float32, trust_remote_code=True)
model.eval()

def ask(q):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=160, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for q in [
    "Our pod payments-api-7c9 in namespace payments was flagged for 'OOM Risk Forecast' at high risk. What's the root cause and how do we remediate it?",
    "How should an SRE handle a 'CPU Throttling' alert?",
]:
    print("\n=== Q:", q)
    print("A:", ask(q), flush=True)
print("\n[verify] DONE — merged model served-ready at", merged)
