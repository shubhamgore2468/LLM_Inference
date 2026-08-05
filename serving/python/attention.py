# paged attention on SM75. Two Paths:
# prefill - ragged, one seq at a time, causal, SDPA over prefix + new tokens. Loops over seq
# decode - batched, one query token per seq, attends over full gathered histpry with length mask

# GQA is handled by repeat interleave on K/V (14 Q Heads / 2 KV heads = 7 x)

import torch
import torch.nn.functional as F 

def _expand_kv(x, n_rep):

    """ (T, kv_headds, D) -> (T, kv_heads * n_rep, D)"""

    if n_rep == 1:
        return x 
    T, H, D = x.shape

    return (x.unsqueeze(2)
            .expand(T, H, n_rep, D)
            .reshape(T, H * n_rep, D))


def paged_attention(
    queries, 
    new_keys, 
    new_values, 
    kv_slab, 
    layer_idx, 
    batch_plan, 
    num_query_heads, 
    num_kv_heads
):
    """Coordinates the execution of prefill or decode paged attention.

    queries, new_keys, new_values : (num_tokens, heads, head_dim) for this iteration
    batch_plan                    : BatchPlan fields as torch tensors / lists
    Returns                       : (num_tokens, num_query_heads, head_dim)
    """
    # Always write first: prefill reads its own KV back for the causal part.
    kv_slab.store(layer_idx, new_keys, new_values, batch_plan["slot_mapping"])
    
    head_repeat_factor = num_query_heads // num_kv_heads
    
    if batch_plan["is_prefill"]:
        return _prefill(queries, kv_slab, layer_idx, batch_plan, head_repeat_factor)
    else:
        return _decode(queries, kv_slab, layer_idx, batch_plan, head_repeat_factor)



def _prefill(queries, kv_slab, layer_idx, batch_plan, head_repeat_factor):

    cum_seq_lens_q = batch_plan["cu_seqlens_q"]   # list[int], len batch_size + 1
    cum_seq_lens_k = batch_plan["cu_seqlens_k"]   # KV length per sequence, including cached prefix
    block_tables = batch_plan["block_tables"]       # (batch_size, max_blocks) int32
    block_size = kv_slab.block_size

    output_tensor = torch.empty_like(queries)
    batch_size = len(cum_seq_lens_q) - 1

    for i in range(batch_size):

        q_start, q_end = cum_seq_lens_q[i], cum_seq_lens_q[i+1]
        num_query_tokens = q_end - q_start
        num_key_tokens = cum_seq_lens_k[i+1] - cum_seq_lens_k[i]

        if num_query_tokens == 0:
            continue

        #Gather blocks for this speific dequence
        num_blocks_needed = (num_key_tokens + block_size - 1) // block_size
        block_ids = block_tables[i, :num_blocks_needed].clamp(min=0).to(torch.long)
        #Flatten linear hardware slots out of block coordinates
        slot_offsets = torch.arange(block_size, device=queries.device)
        slot_mappings = (block_ids.unsqueeze(-1) * block_size + slot_offsets).reshape(-1)[:num_key_tokens]

        #Retrieve non continguous data slices out of physical storage
        keys = kv_slab.cache[0, layer_idx].index_select(0, slot_mappings)
        values = kv_slab.cache[1, layer_idx].index_select(0, slot_mappings)

        keys = _expand_kv(keys, head_repeat_factor)

        values = _expand_kv(values, head_repeat_factor)

        #Prepare shaped for SDPA: (batch=1, heas, seq_len, head_dim)
        query_slice = queries[q_start:q_end].transpose(0, 1).unsqueeze(0)
        key_slice = keys.transpose(0, 1).unsqueeze(0)
        value_slice = values.transpose(0, 1).unsqueeze(0)

        #The prefix-cache: we compute num_query_tokens queries but atten over num_key_tokens
        #keys, where num_key_tokens >= num_query_tokens because a cache hit 
        # chunk already contributed KV. Query j sits at absollute position
        # (num_key_tokens - num_query_tokens + j), so causal mask must beOFFSET not triangular

        causal_offset = num_key_tokens - num_query_tokens
        query_indices = torch.arange(num_query_tokens, device=queries.device).unsqueeze(1) + causal_offset
        key_indices = torch.arange(num_key_tokens, device=queries.device).unsqueeze(0)
        attention_mask = (key_indices <= query_indices)

        computed_attention = F.scaled_dot_product_attention(
                query_slice, key_slice, value_slice, attn_mask=attention_mask
                )
        output_tensor[q_start:q_end] = computed_attention.squeeze(0).transpose(0, 1)

    return output_tensor

def _decode(queries, kv_slab, layer_idx, batch_plan, head_repeat_factor):
    
    """ Process uniform individual-token lookups concurrently using a layout mask"""

    block_tables = batch_plan["block_tables"]
    context_lengths = batch_plan["context_lens"]
    bsz = block_tables.shape[0]

    keys_gathered, values_gathered = kv_slab.gather(layer_idx, block_tables, context_lengths)

    max_history_length = keys_gathered.shape[1]

    def format_and_expand_kv(kv_tensor):

        bsz, seq_len, kv_h, h_dim = kv_tensor.shape
        expanded = (
                kv_tensor.unsqueeze(3)
                .expand(bsz, seq_len, kv_h, head_repeat_factor, h_dim)
                .reshape(bsz, seq_len, kv_h * head_repeat_factor, h_dim)
                )
        return expanded.transpose(1, 2) # -> (bsz, q_heads, max_history_len, head_dim)

    keys_formatted = format_and_expand_kv(keys_gathered)
    values_formatted = format_and_expand_kv(values_gathered)

    #Reshape queries from flat to bsz, q_heads, 1, head_dim
    head_dim = queries.shape[-1]
    queries_formatted = queries.view(bsz, 1, -1, head_dim).transpose(1, 2)

    # Mask everything past each seq's real length. The gather returns whole blocks
    # so tail of last block is uninitialized memory

    timeline_positions = torch.arange(max_history_length, device=queries.device).unsqueeze(0)
    attention_mask = (timeline_positions < context_lengths.unsqueeze(1)).view(bsz, 1, 1, max_history_length)

    computed_attention = F.scaled_dot_product_attention(
            queries_formatted, keys_formatted, values_formatted, attn_mask=attention_mask
            )

    return computed_attention.transpose(1, 2).reshape(bsz, -1, head_dim)
