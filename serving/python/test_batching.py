import argparse
import torch
from transformers import AutoTokenizer

from runner import Engine

def gen(engine, ids, n=32, ignore_eos=True):
    sid = engine.add(ids, max_tokens=n, ignore_eos=ignore_eos)

    while engine.sched.has_work():

        engine.step()

    return engine.tokens_of(sid)[len(ids):]

def main():

    p  = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--fp32", action='store_true')

    args = p.parse_args()

    dtype = torch.float32 if args.fp32 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)

    prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "In a shocking finding, scientists discovered",
            ]

    ids = [tok(x).input_ids for x in prompts]

    #--------------baseline one at a time--------------
    e = Engine(args.model, dtype=dtype, kv_gb=2.0)
    solo = [gen(e, i) for i in ids]

    print("baseline captured")

    #------------------test 1: concurrent must equal solo--------------
    print("\n=== batched == solo ====")
    e = Engine(args.model, dtype=dtype, kv_gb=2.0)
    sids = [e.add(i, max_tokens=32, ignore_eos=True) for i in ids]

    while e.sched.has_work():
        e.step()

    for k, sid in enumerate(sids):
        out = e.tokens_of(sid)[len(ids[k]):]

        assert out == solo[k], f"seq {k} diverged when batched"

        print(f" OK {prompts[k][:32]!r}")


    #----------test 2: prefix sharing--------
    # 3 reqs behind one long sys prompt, 2nd and 3rd should skip prefill
    print("Prefix Cache")

    e = Engine(args.model, dtype=dtype, kv_gb=2.0)
    shared = tok("You are a helpful assistant. " * 40).input_ids

    for i, suffix in enumerate(["What is 2+2?", "Name a color", "Say Hello"]):
        full = shared + tok(suffix).input_ids

        sid = e.add(full, max_tokens=8, ignore_eos=True)

        while e.sched.has_work():
            e.step()

        stats = e.stats()

        total = stats.prefix_hit_blocks + stats.prefill_blocks

        print(f"  hit blocks={stats.prefix_hit_blocks} computed={stats.prefill_blocks} "
          f"hit_rate={stats.prefix_hit_blocks / max(total,1):.1%}")

        # First request populates the cache — 0 hits is correct there.
        # Only 2nd/3rd requests, sharing the same prefix, should hit.
        if i > 0:
            assert stats.prefix_hit_blocks > 0, "no prefix reuse — check seal_blocks / hash chain"
 
    #------------test 3: chunked prefill ------

    print("\n ------ chunked perfill ------")

    e = Engine(args.model, dtype=dtype, kv_gb=2.0, chunk_size=8)

    out = gen(e, ids[0])

    assert out == solo[0], "chunked prefill changed output"
    print(" OK output unchanged with chunk_size = 8")


    # ------------ test 4: preemption -------------

    e = Engine(args.model, dtype=dtype, kv_gb=2.0, max_kv_utilization=0.5)

    sids = [e.add(i, max_tokens=32, ignore_eos=True) for i in ids]

    while e.sched.has_work():
        e.step()
    stats = e.stats()

    print(f"  preempted {stats.total_preempted} times")
    for k, sid in enumerate(sids):
        out = e.tokens_of(sid)[len(ids[k]):]
        assert out == solo[k], f"seq {k} corrupted by preemption"
    assert stats.total_preempted > 0, "watermark never tripped — lower kv_gb"
    print("  OK  output survives preemption")
 
    print("\nall batching checks passed")
 
 
if __name__ == "__main__":
    main()
    
