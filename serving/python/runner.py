# The engine loop. C++ decides what to run and this turns into tensors, runs fwd pass, samples and gives tokens back

#plan = sched.step()  - what to compute
#forward(plan) - hidden states
#sample - one token per sequence
#sched.commit(tokens) - append, allocate, seal hashes, peempt


import torch

import _kvsched as ks
from kvslab import KVSlab
from loader import load_model
from typing import List, Tuple, Dict, Any

class Engine:
    """
    Manages the execution of LM inference, including KV cache memory allocation, batch scheduling and token generation steps.
    """

    def __init__(
            self, 
            model_path, 
            device="cuda", 
            dtype=torch.float16, 
            kv_gb=8.0, 
            block_size=16, 
            **sched_kwargs):

        self.model, self.cfg = load_model(model_path, device=device, dtype=dtype)

        self.device = device
        head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
        bytes_per_token = (2 * self.cfg.num_hidden_layers *
                           self.cfg.num_key_value_heads * head_dim *
                           torch.empty(0, dtype=dtype).element_size()) # KV cache size for single token, 2* for KV vectors

        n_blocks = int(kv_gb * 1024**3 // (bytes_per_token * block_size))

        self.slab = KVSlab(self.cfg.num_hidden_layers, n_blocks, block_size, 
                           self.cfg.num_key_value_heads, head_dim, dtype=dtype, device=device)

        cfg = ks.SchedulerConfig()
        cfg.block_size = block_size
        
        for k, v in sched_kwargs.items():
            setattr(cfg, k, v)

        self.bm = ks.BlockManager(n_blocks, block_size)

        self.sched = ks.Scheduler(self.bm, cfg)

        print(f"[engine] {n_blocks} blocks x {block_size} tok "
              f"= {n_blocks * block_size:,} tokens "
              f"({bytes_per_token / 1024:.1f} KB/token)")

    def add(
            self, 
            token_ids, 
            max_tokens=128, 
            ignore_eos=False, 
            arrival_ts=0.0
            ):
        """
        Converts the CPU/numpy scheduling plan into pytorch tensors in the target device.
        """
        return self.sched.add(list(token_ids), max_tokens, self.cfg.eos_token_id, ignore_eos, arrival_ts)


    def _prepare_torch_inputs(self, plan: Any) -> Dict[str, Any]:
        """Converts the CPU/numpy scheduling plan into PyTorch tensors on the target device."""
        
        def to_tensor(array: Any, target_dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(array).to(
                device=self.device, 
                dtype=target_dtype, 
                non_blocking=True
            )

        return {
            "is_prefill": plan.is_prefill,
            "token_ids": to_tensor(plan.token_ids, torch.long),
            "positions": to_tensor(plan.positions, torch.long),
            "slot_mapping": to_tensor(plan.slot_mapping, torch.long),
            "context_lens": to_tensor(plan.context_lens, torch.int32),
            "block_tables": to_tensor(plan.block_tables, torch.int32).view(plan.n_seqs(), plan.max_blocks),
            "cu_seqlens_q": list(plan.cu_seqlens_q),
            "cu_seqlens_k": list(plan.cu_seqlens_k),
        }


    @torch.no_grad()

    @torch.no_grad()
    def step(self) -> List[Tuple[int, int]]:
        """Executes one iteration of inference. Returns a list of (sequence_id, new_token_id)."""
        
        plan = self.sched.step() # Fetch the next batch plan
        
        if plan.n_seqs() == 0:
            return []

        model_inputs = self._prepare_torch_inputs(plan)

        # Forward pass through the model
        hidden_states = self.model(model_inputs, self.slab)

        # Extract the hidden states representing the *last* token of each sequence
        query_seq_lens = model_inputs["cu_seqlens_q"]
        last_token_indices = torch.tensor(
            [query_seq_lens[i + 1] - 1 for i in range(plan.n_seqs())], 
            device=self.device
        )

        # Compute logits and pick the most likely next token (greedy decoding)
        logits = self.model.logits(hidden_states[last_token_indices])
        next_tokens = logits.argmax(dim=-1).tolist()

        # Update the scheduler with the newly generated tokens
        self.sched.commit(next_tokens)

        # Zip together the sequence IDs and their respective new tokens
        sequence_ids = plan.seq_ids.tolist()
        return list(zip(sequence_ids, next_tokens))  


    def run_until_done(self):
        while self.sched.has_work():
            self.step()
        return self.sched.take_finished()

    def stats(self):
        return self.sched.stats()

    def tokens_of(self, seq_id):
        return self.sched.tokens_of(seq_id)
