import argparse, asyncio, json, random, time, httpx
from metrics import HDR, row, summarize

async def one(client, url, prompt, osl, out):
    t0 = time.perf_counter()
    first = None; n = 0

    body = {"model": "m", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": osl, "stream": True}

    async with client.stream("POST", url, json=body, timeout=600) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "): continue

            if line[6:].strip() == "[DONE]": break

            if json.loads(line[6:])["choices"][0]["delta"].get("content"):
                n += 1
                if first is None: first = time.perf_counter()

    out.append({"arrival": t0, "first": first, "done": time.perf_counter(), "tokens": n})


async def closed(client, url, prompt, osl, c, reqs):
    out = []

    async def worker():
        for _ in range(reqs // c): await one(client, url, prompt, osl, out)

    await asyncio.gather(*[worker() for _ in range(c)])

    return out


async def open_loop(client, url, prompt, osl, rate, reqs):

    out, tasks = [], []

    for _ in range(reqs):
        tasks.append(asyncio.create_task(one(client, url, prompt, osl, out)))
        await asyncio.sleep(random.expovariate(rate))

    await asyncio.gather(*tasks)

    return out

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--concurrency", default="1,4,16")
    p.add_argument("--rate", type=float)          # set this for open-loop instead
    p.add_argument("--isl", type=int, default=512)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--reqs", type=int, default=32)
    p.add_argument("--out", default="benchmarks/baseline_hf.md")
    a = p.parse_args()
    prompt = "the quick brown fox jumps over the lazy dog " * (a.isl // 9)  # ~isl tokens
    rows = []
    async with httpx.AsyncClient() as c:
        await one(c, a.url, "warmup", 8, [])     # discarded
        if a.rate:
            rows.append(summarize(await open_loop(c, a.url, prompt, a.osl, a.rate, a.reqs),
                                  f"open λ={a.rate}"))
        else:
            for n in [int(x) for x in a.concurrency.split(",")]:
                rows.append(summarize(await closed(c, a.url, prompt, a.osl, n, max(a.reqs, n)),
                                      f"closed c={n}"))
    md = f"# baseline_hf (isl={a.isl} osl={a.osl})\n\n{HDR}\n" + "\n".join(row(r) for r in rows) + "\n"
    open(a.out, "a").write(md); print(md)

asyncio.run(main())
