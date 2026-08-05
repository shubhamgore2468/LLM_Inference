import torch

class KVSlab:

    def __init__(self, n_layers, n_blocks, block_size, kv_heads, head_dim, dtype=torch.float16, device="cuda"):

        self.n_layers = n_layers
        self.block_size = block_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim


        # Flat in (Block, slot) so slot mapping is a single linear index
        # Layout: [2, L, n_blocks * block_size, H, D]

        self.cache = torch.zeros(
                2, n_layers, n_blocks * block_size, kv_heads, head_dim, dtype=dtype, device=device
                )

    def bytes_per_token(self):

        return 2 * self.n_layers * self.kv_heads * self.head_dim * self.cache.element_size()


    def store(self, layer, k, v, slot_mapping):
        """
        Scatter this iteerations K/V into their blocks.
        k, v: (n_tokens, kv_heads, head_dim)
        slot_mapping: (n_tokens, ) int64, = block_id * block_size + offset


        """
        self.cache[0, layer].index_copy_(0, slot_mapping, k)

        self.cache[1, layer].index_copy_(0, slot_mapping, v)

    def gather(self, layer, block_tables):
        """
        Read back full KV histories for a batch

        block_tables: (B, max_blocks) int32, -1 padded
        context_lens: (B, ) int32, true KV length per sequence

        Returns k, v of shape (B, max_Blocks * block_size, H, D), with garbage
        beyond context_lens - the caller must mask
        """

        B, max_blocks = block_tables.shape
        bs = self.block_size

        # -1 padding would index from the end of slab; clamp to 0 and lets caller mask discard those positions

        safe = block_tables.clamp(min=0).to(torch.long)

        # (B, max_blocks, bs) linear slots indices

        offs = torch.arange(bs, device=safe.device)

        slots = (safe.unsqueeze(-1) * bs + offs).reshape(B, -1)

        k = self.cache[0, layer].index_select(0, slots.reshape(-1))

        v = self.cache[1, layer].index_select(0, slots.reshape(-1))

        k = k.view(B, max_blocks * bs, self.kv_heads, self.head_dim)

        v = v.view(B, max_blocks * bs, self.kv_heads, self.head_dim)

        return k, v
