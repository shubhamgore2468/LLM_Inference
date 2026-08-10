#pragma once

#include <cstdint>
#include <deque>
#include<memory>
#include<unordered_map>
#include<vector>

#include "block.h"
#include "sequence.h"

namespace kvsched{

  struct SchedulerConfig{
    int32_t block_size = 16;
    int32_t max_batch_seqs = 64; //how many seqs acn be processed concurrently
    int32_t max_batch_tokens = 2048; //total token budget for single fwd pass
    int32_t chunk_size = 512; // chunked prefill
    double max_kv_utilization = 0.95; // kv cache util 95%, preempt(pause and swap out)
  };

  struct Stats {
    int32_t waiting = 0;
    int32_t running = 0;
    int32_t in_flight_tokens = 0;
    int64_t total_finished = 0;
    int64_t total_preempted = 0;
    int64_t prefix_hit_blocks = 0;
    int64_t prefill_blocks = 0;
    double kv_util = 0.0;
  };

  class Scheduler {
    public: 
      Scheduler(BlockManager& bm, SchedulerConfig cfg);

      uint64_t add(std::vector<int32_t> tokens, int32_t max_tokens, int32_t eos, bool ignore_eos, double arrival_ts);

      
      BatchPlan step();//Build one iterations batch, Empty plan (n_seqs()==0)

      //Next_tokens is parallel to plan.seq_ids one sampled token per sequence
      //
      //For sequences still in prefill the value is ignored
      //
      void commit(const std::vector<int32_t>& next_tokens);

      bool has_work() const { return !waiting_.empty() || !running_.empty(); }

      Stats stats() const;

      //Completed sequences, drained by caller each iterations
      //
      std::vector<uint64_t> take_finished();

      const std::vector<int32_t>& tokens_of(uint64_t id) const;

    private:

      int32_t match_prefix(Sequence& s);
      bool try_admit(Sequence& s, int32_t token_budget);
      void seal_blocks(Sequence& s);
      void preempt(size_t idx);
      void release(Sequence& s);

      BlockManager& bm_;
      SchedulerConfig cfg_;
      uint64_t next_id_ = 1;


      std::deque<Sequence*> waiting_;
      std::vector<Sequence*> running_;
      std::unordered_map<uint64_t, std::unique_ptr<Sequence>> seqs_;
      std::vector<uint64_t>finished_;

      Stats stats_;
  };

}
