def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); return xs[min(int(round(p / 100 * (len(xs) - 1))), len(xs) - 1)]

def summarize(recs, label):
    ttft = [r["first"] - r["arrival"] for r in recs if r["first"]]
    tpot = [(r["done"] - r["first"]) / (r["tokens"] - 1) for r in recs if r["tokens"] > 1]
    wall = max(r["done"] for r in recs) - min(r["arrival"] for r in recs)
    return dict(label=label, n=len(recs),
                ttft_p50=pct(ttft, 50), ttft_p99=pct(ttft, 99),
                tpot_p50=pct(tpot, 50), tpot_p99=pct(tpot, 99),
                out_tps=sum(r["tokens"] for r in recs) / wall)

HDR = "| run | n | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | out tok/s |\n|---|---|---|---|---|---|---|"
def row(s):
    return "| {label} | {n} | {ttft_p50:.3f} | {ttft_p99:.3f} | {tpot_p50:.4f} | {tpot_p99:.4f} | {out_tps:.1f} |".format(**s)
