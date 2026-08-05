import threading, json, torch, asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

def pick_device(force_fp32=False):
    if torch.cuda.is_available():
        return "cuda", torch.float32 if force_fp32 else torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32 if force_fp32 else torch.float16
    return "cpu", torch.float32

DEVICE, DTYPE = pick_device()

print(DEVICE, DTYPE)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
model.eval()

app = FastAPI()
lock = threading.Lock()

@app.post("/v1/chat/completions")
async def chat(req: dict):
    n = req.get("max_tokens", 128)
    
    ids = tokenizer.apply_chat_template(
        req["messages"], 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(DEVICE)
    
    st = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Define generation task
    def run_generation():
        try:
            with lock:
                model.generate( **ids, streamer=st, max_new_tokens=n, do_sample=False)
        finally:
            st.end()
    threading.Thread(target=run_generation, daemon=True).start()

    async def gen():
        loop = asyncio.get_event_loop()
        it = iter(st)

        while True:
            t = await loop.run_in_executor(None, lambda: next(it, None))
            if t is None: break
            if t: yield f'data: {json.dumps({"choices":[{"delta":{"content":t}}]})}\n\n'
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
