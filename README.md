24 layers, hidden_size = 896, 14 attn_heads, 896/14 = 64 wide = head_dim

2 KV heads 14 query heads, GQA, 7:1.14

bfloat16, 32K context, vocab 152K



kv_bytes_per_token = 2 x 24 x 2 x 64 x 2 = 12, 288 = 12
kv_bytes_per_tokenbytes_per_block = 12,288 x 16 = 192 kv_bytes_per_token

on T4, 
n_blocks = 13GB / 192KB = 71000 blocks = 1.14M
                        277 concurrent sequences at 4K context
                        1100 at 1kv_bytes_per_token

