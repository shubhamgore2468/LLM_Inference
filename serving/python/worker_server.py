import argparse
import asyncio
import json
import os
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI 
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from engine import EngineThread

app = FastAPI()

ENGINE: EngineThread = None
TOK = None
READY = False
WORKER_ID = os.environ.get("WORKER_ID", "w0")

class GenRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
    stream: bool = True


def _encode(req: GenRequest):
    if req.messages:
        ids = TOK.apply_chat_template(req.messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return list(ids)
    return list(TOK(req.prompt or "").input_ids)

async def _token_stream(ids, req: GenRequest):
    """ Yield (delta_text, n_tokens) per engine event."""

    loop = asyncio.get_running_loop()
    req_queue: asyncio.Queue = asyncio.Queue()

    def on_event(event):
        """
        Engine thread asyncio.Queue is NOT thread safe, so hand off through the loop
        """

        loop.call_soon_threadsafe(req_queue.put_nowait, event)

    await asyncio.to_thread(
                ENGINE.submit_request, ids, req.max_tokens, req.ignore_eos, on_event
                )

    out, sent = [], ""

    while True:
        event = await req_queue.get()
        if event.get("done"):
            return
        out.append(event["token"])
            
        full = TOK.decode(out)
        delta, sent = full[len(sent):], full
        yield delta, 1

@app.post("/generate")
async def generate(req: GenRequest):
    ids = _encode(req)

    if not req.stream:
        text, n = "", 0

        async for d, k in _token_stream(ids, req):
            text += d
            n += k

        return {"text" : text, "n_tokens" : n}

    async def sse():
        n = 0

        async for delta, k in _token_stream(ids, req):
            n += k

            yield f"data: {json.dumps({'text': delta, 'n_tokens': k})}\n\n"
        yield f"data: {json.dumps({'done': True, 'n_tokens': n})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"}
            )

@app.post("/v1/chat/completions")
async def chat(req: GenRequest):

    ids = _encode(req)
    rid = f"chatcompletion-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if not req.stream:
        text, n = "", 0

        async for d, k in _token_stream(ids, req):
            text += d
            n += k

        return {
           "id": rid, "object": "chat.completion", "created": created,
            "model": WORKER_ID,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": n,"total_tokens": len(ids) + n},
           }

    async def sse():
        base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": WORKER_ID}

        async for delta, k in _token_stream(ids, req):

            chunk = {**base, "choices" : [{"index" : 0, "delta": {"content":delta}}], "n_tokens":k}
            yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
            sse(), media_type="text/event-stream", 
            headers={"Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"}
    )

@app.get("/stats")
async def stats():
    s = ENGINE.get_metrics()
    s["worker_id"] = WORKER_ID
    s["ts"] = time.time()
    return s
 
 
@app.get("/health/live")
async def live():
    # Process is up. Failing this means "restart me".
    return {"ok": True}

@app.get("/health/ready")
async def ready():
    # Failing this means "don't route to me" — a very different instruction.
    # Conflating the two causes cascading restarts: a pod slow to warm up gets
    # killed by its liveness probe, restarts, is slow again, forever.
    if not READY:
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True}

def main():

    global ENGINE, TOK, READY
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--kv-gb", type=float, default=8.0)
    p.add_argument("--max-batch-tokens", type=int, default=2048)
    p.add_argument("--max-batch-seqs", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()

    TOK = AutoTokenizer.from_pretrained(args.model)
    ENGINE = EngineThread(
            args.model,
            dtype=torch.float32 if args.fp32 else torch.float16,
            kv_gb=args.kv_gb,
            max_batch_tokens=args.max_batch_tokens,
            max_batch_seqs=args.max_batch_seqs,
            chunk_size=args.chunk_size,
            )
    done = threading.Event()
    ENGINE.submit_request(TOK("warmup").input_ids, 4, True, lambda event: event.get("done") and done.set())
    if not done.wait(timeout=120):
        print(f"[worker] {WORKER_ID} warmup timed out after 120s — starting anyway")
    READY = True
    print(f"[worker] {WORKER_ID} ready on: {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
