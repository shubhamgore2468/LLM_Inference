# py/test_parity.py
#
# Phase 2 acceptance. Two tests, in this order:
#
#   1. one-shot prefill logits vs HF  — isolates layer math from KV paging
#   2. multi-step paged decode vs HF greedy — exercises blocks, slot_mapping,
#      and the offset causal mask
#
# If (1) passes and (2) fails, the bug is in paging, not in the model. That
# split is the whole point of testing them separately.
#
# Run: python py/test_parity.py --model ~/models/Qwen2.5-0.5B-Instruct

import argparse
import torch
from bench.server_hf import pick_device 
from kvslab import KVSlab
from loader import load_model

BLOCK = 16


def make_plan(tokens, block_ids, n_cached, device):
    """Minimal hand-built BatchPlan. In Phase 3 the C++ scheduler emits this."""
    n_new = len(tokens) - n_cached
    new = tokens[n_cached:]
    slots = [block_ids[(n_cached + i) // BLOCK] * BLOCK + (n_cached + i) % BLOCK
             for i in range(n_new)]
    return {
        "is_prefill": n_new > 1,
        "token_ids": torch.tensor(new, device=device),
        "positions": torch.arange(n_cached, len(tokens), device=device),
        "slot_mapping": torch.tensor(slots, device=device, dtype=torch.long),
        "cu_seqlens_q": [0, n_new],
        "cu_seqlens_k": [0, len(tokens)],
        "block_tables": torch.tensor([block_ids], device=device, dtype=torch.int32),
        "context_lens": torch.tensor([len(tokens)], device=device, dtype=torch.int32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    dev, dtype = pick_device(args.fp32)    
    model, cfg = load_model(args.model, device=dev, dtype=dtype)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    slab = KVSlab(cfg.num_hidden_layers, 256, BLOCK,
                  cfg.num_key_value_heads, head_dim, device=dev, dtype=dtype)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=dev).eval()

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In a shocking finding, scientists discovered a herd of unicorns",
    ]

    # ---- test 1: prefill logits ----
    print("=== prefill logits vs HF ===")
    for prompt in prompts:
        ids = tok(prompt).input_ids
        nb = (len(ids) + BLOCK - 1) // BLOCK
        blocks = list(range(nb))

        with torch.no_grad():
            plan = make_plan(ids, blocks, 0, dev)
            h = model(plan, slab)
            mine = model.logits(h[-1])
            ref = hf(torch.tensor([ids], device=dev)).logits[0, -1]

        diff = (mine.float() - ref.float()).abs().max().item()
        agree = mine.argmax().item() == ref.argmax().item()
        print(f"  max|Δ|={diff:.4f}  argmax_match={agree}  {prompt[:32]!r}")
        assert agree, "prefill argmax mismatch — check RoPE layout, QKV bias, rope_theta"

    # ---- test 2: paged greedy decode ----
    print(f"\n=== {args.steps}-step paged decode vs HF greedy ===")
    for prompt in prompts:
        ids = tok(prompt).input_ids
        blocks = list(range(64))
        slab.cache.zero_()

        seq = list(ids)
        with torch.no_grad():
            plan = make_plan(seq, blocks, 0, dev)
            h = model(plan, slab)
            mine_logits = model.logits(h[-1])
            nxt = mine_logits.argmax().item()
            if args.debug:
                ref_logits = hf(torch.tensor([seq], device=dev)).logits[0, -1]
                d = (mine_logits.float() - ref_logits.float()).abs().max().item()
                print(f"    step 0  max|Δ|={d:.4f}")
            seq.append(nxt)

            for step in range(args.steps - 1):
                plan = make_plan(seq, blocks, len(seq) - 1, dev)
                h = model(plan, slab)
                mine_logits = model.logits(h[-1])
                nxt = mine_logits.argmax().item()
                if args.debug:
                    ref_logits = hf(torch.tensor([seq], device=dev)).logits[0, -1]
                    d = (mine_logits.float() - ref_logits.float()).abs().max().item()
                    print(f"    step {step + 1}  max|Δ|={d:.4f}")
                seq.append(nxt)

        ref = hf.generate(torch.tensor([ids], device=dev),
                          max_new_tokens=args.steps, do_sample=False,
                          pad_token_id=tok.eos_token_id)[0].tolist()

        mine_out, ref_out = seq[len(ids):], ref[len(ids):]
        if mine_out == ref_out:
            print(f"  OK  {prompt[:32]!r}")
        else:
            n = next(i for i, (a, b) in enumerate(zip(mine_out, ref_out)) if a != b)
            print(f"  DIVERGED at token {n}  {prompt[:32]!r}")
            print(f"    mine: {tok.decode(mine_out[:n+3])!r}")
            print(f"    ref : {tok.decode(ref_out[:n+3])!r}")
            raise SystemExit(1)

    print("\nall parity checks passed")


if __name__ == "__main__":
    main()
