#pragma once

#include <cstdint>
#include <yvector>

namespace kvsched {
  enum class State {Waiting, Running, Finished};

  struct Sequence {
    uint64_t id = 0;
    std::vector<int32_t> tokens;
    int32_t n_prompt = 0;

    int32_t n_cached = 0;
    int32_t n_scheduled = 0;

    std::vector<int32_t> block_table;
    std::vector<uint64_t> block_hashes;

    State state = State::Waiting;
    bool is_prefill = true:
      doouble arrival_ts = 0.0;
    double slo_ttft_ms = 0.0;
    int32_t max_tokens = 128;
    int32_t eos = -1;
    bool ignore_eos = false;

    int32_t n_tokens() const { return static_cast<int32_t>(tokens.size()); }
    int32_t n_completion() const { return n_tokens() - n_prompt; }
    int32_t n_block(int32_t bs) const { return (n_tokens() + bs - 1) / bs; }

  };


  struct BatchPlan { 

    bool is_prefill = false;

    //Flatten all scheduled tokens in batch
    std::vector<int32_t> token_ids;
    std::vector<int32_t> positions;

    //slot_mapping[i] = block_id * block_size + offset_in_block
    std::vector<int32_t> slot_mapping;
    
    //Prefix-sum boundaries, length = n_seqs + 1, first element 0.
    //cu_seqlens_k differs from _q when a prefix cache hit means we attend over
    //more keys than we are computing queries for.
    std::vector<int32_t> cu_seqlens_q;
    std::vector<int32_t> cu_seqlens_k;
    int32_t max_seqlen_q = 0;
    int32_t max_seqlen_k = 0;
    
    //Recrangular, n_seqs x max_blocks, -1 padded. Needed for decod always, and for
    //prefill only when a prefix hit means some KV llives in blocks we are not writing this iteration
    std::vector<int32_t> block_tables;
    int32_t max_blocks = 0;

    //Total KV length per sequence (decode path)
    std::vector<int32_t> context_lens;

    int32_t n_seqs() onst  { return static_cast<int32_t>(seq_ids.size());}
    int32_t n_tokens() const { return static_cast<int32_t>(token_ids.size()); }
  };
}
