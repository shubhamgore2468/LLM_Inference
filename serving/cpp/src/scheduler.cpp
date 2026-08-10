#include "scheduler.h"

#include <algorithm>
#include <cassert>

#include "hash.h"

namespace kvsched {

  Scheduler::Scheduler(BlockManager & bm, SchedulerConfig cfg) : bm_(bm), cfg_(cfg){}

  
  /**
   * Entry point for new requests. wraps new tokens into sequence object, 
   * assigns incrementing id, sets to waiting and pushed to waiting_queue
   **/
  uint64_t Scheduler::add(std::vector<int32_t> tokens, int32_t max_tokens, int32_t eos, bool ignore_eos, double arrival_ts){

    auto s = std::make_unique<Sequence>();

    s->id = next_id_++;
    s->n_prompt = static_cast<int32_t>(tokens.size());
    s->tokens = std::move(tokens);
    s->max_tokens = max_tokens;
    s->eos = eos;
    s->ignore_eos = ignore_eos;
    s->arrival_ts = arrival_ts;
    s->state = State::Waiting;

    uint64_t id = s->id;
    waiting_.push_back(s.get());
    seqs_[id] = std::move(s);
    return id;

  }

  //Walk the block-hash chain against the cache. Stop at first miss
  //Block i is only resuable if every block before it matched - PREFIX cache
  //

  int32_t Scheduler::match_prefix(Sequence& s){

    auto chain = hash_chain(s.tokens, cfg_.block_size); // generated sequence of hashes for tokerns, chunked by block_size

    int32_t matched = 0;

    for(uint64_t h: chain) {
      auto hit = bm_.lookup(h);
      if (!hit) break;
      bm_.incref(*hit);
      s.block_table.push_back(*hit);
      s.block_hashes.push_back(h);
      ++matched;
    }
    // Never let the whole prompt be cached; we need atleast one token for fwd pass.
    // Forward pass to produce logits for the next tokens
    //
    int32_t cached = matched * cfg_.block_size; // convert total number of matched blocks into total number of matched tokens

    if (cached >= s.n_prompt) { // check if prefix cache covered entire prompt
      --matched; // dec lastblock, run forward pass on that to generate token 
      bm_.decref(s.block_table.back());
      s.block_table.pop_back();
      s.block_hashes.pop_back();
      cached = matched * cfg_.block_size; // update tokens
    }

    s.n_cached = cached;
    stats_.prefix_hit_blocks += matched;

    return cached;
  }
  

  /**
   * evals if waiting seq can be moved to active running waiting_queue
   * runs prefix match to to see how mych prompt canbe skipped
   * cals how many tokens need to be computed and chunk size
   * cals how many phy blocks needed
   * allocs, sets schd and state = running
   **/

  bool Scheduler::try_admit(Sequence& s, int32_t token_budget) {

    match_prefix(s); //check prefix cache

    //                          total remaining token in prompt, hard limit chunk size, budget entire batch
    int32_t to_compute = std::min({s.n_prompt - s.n_cached, cfg_.chunk_size, token_budget});

    if(to_compute <= 0) return false;

    //Blocks needed to hold n_cached + to compute tokens -  prefix hit
    //
    int32_t need_total = (s.n_cached + to_compute + cfg_.block_size - 1)  / cfg_.block_size;

    int32_t need = need_total - static_cast<int32_t>(s.block_table.size());

    if (need > 0){

      auto got = bm_.allocate(need);

      if(!got) { // OOM handhling, seq stays in waiting q until memory frees up
        release(s);
        return false;
      }

      //successfull alloc, iterate phy block, add to seq page table
      for(int32_t id: *got){
        s.block_table.push_back(id);
        s.block_hashes.push_back(0);
      }
      stats_.prefill_blocks += need;
    }

    s.n_scheduled = to_compute;
    s.is_prefill = true;
    s.state = State::Running;
    return true;

  }

  BatchPlan Scheduler::step() 
  {

    BatchPlan plan;
    const int32_t bs = cfg_.block_size;

    // ---------- 1. reap finished ---------
    for( auto it = running_.begin(); it != running_.end();){
      if ((*it) -> state == State::Finished) it = running_.erase(it);
      else ++it;
    }

    // ---------- 2. Schedule running decode first--------
    // decodes are already admitted hold their blocks - priority
    // starving for a prefill - ITL spike
    
    int32_t budget = cfg_.max_batch_tokens;

    std::vector<Sequence*> batch;

    for (Sequence * s: running_) { // iterate running queue

      if ( s->is_prefill) continue;

      if(static_cast<int32_t>(batch.size()) >= cfg_.max_batch_seqs) break;

      if(budget < 1 ) break;

      //growing into a new block

      int32_t need = (s->n_tokens() + bs - 1) / bs;


      if( need > static_cast<int32_t> (s->block_table.size())) {
        auto got = bm_.allocate(1);

        if(!got) {
          s->n_cached = 0;
          s->is_prefill = true;
          s->state = State::Waiting;
          release(*s);
          waiting_.push_back(s);
          continue;
        }

        s-> block_table.push_back((*got)[0]);
        s->block_hashes.push_back(0);
      }
      
      s->n_scheduled = 1;
      batch.push_back(s);
      budget -= 1;
    }

    // ----- 3. continue in-flight prefilss, then admit new ones as decodes done
    //
    for ( Sequence * s: running_) {

      if(!s->is_prefill) continue;

      if(static_cast<int32_t>(batch.size()) >= cfg_.max_batch_seqs) break;

      if(budget <= 0) break;

      //How many tokens of prompt this seq is allowed to process this iteration.
      int32_t to_compute = std::min({s->n_prompt - s->n_cached, cfg_.chunk_size, budget});

      if(to_compute <= 0) continue;

      // calcs if chunks of tokens need a new block or fit within this block boundary
      int32_t need_total = (s->n_cached + to_compute + bs - 1) / bs;
      int32_t need = need_total - static_cast<int32_t>(s->block_table.size());

      if(need > 0) {
        auto got = bm_.allocate(need);
        if(!got) continue;

        for(int32_t id: *got){ // if memory granted, map phy blocks to seqs
          s->block_table.push_back(id);
          s->block_hashes.push_back(0);
        }
      }
      s->n_scheduled = to_compute;
      batch.push_back(s);
      budget -= to_compute;
    }
    
    // clear seqs from waiting queue
    while (!waiting_.empty() && budget > 0  && static_cast<int32_t>(batch.size()) < cfg_.max_batch_seqs){
      Sequence* s = waiting_.front();

      if(!try_admit(*s, budget)) break; // if no resources break out
      // else push to running queue
      waiting_.pop_front();
      running_.push_back(s);
      batch.push_back(s);
      budget -= s->n_scheduled;
    }

    if(batch.empty()) return plan;

    //---- 4. Materialize the plan ----------------
    //A batch is prefill if any member is computing more than one token
    //Mixed bathc take prefill path, which handles n_q == 1 
    
    // if plan has more than one token is_prefill TRUE or else decode, is_prefill FALSE
    plan.is_prefill = std::any_of(batch.begin(), batch.end(), [](Sequence* s) { return s->n_scheduled > 1;});

    plan.cu_seqlens_q.push_back(0);
    plan.cu_seqlens_k.push_back(0);

    size_t max_blocks = 0;
    // GPU requires fixed shaped tensors, so calc MAX SIZE TENSOR
    for(Sequence* s: batch)
      max_blocks = std::max(max_blocks, s -> block_table.size());

    plan.max_blocks = static_cast<int32_t>(max_blocks);

    for ( Sequence* s: batch) {
      plan.seq_ids.push_back(s->id);

      int32_t start = s->n_cached;
      int32_t end = s->n_cached + s->n_scheduled;

      for(int32_t p = start; p < end; ++p){
        plan.token_ids.push_back(s->tokens[static_cast<size_t>(p)]);
        plan.positions.push_back(p);
        int32_t blk = s->block_table[static_cast<size_t>(p / bs)]; // p / bs = tokens position / block_size
        plan.slot_mapping.push_back(blk * bs + ( p % bs));
        // blk * bs + (p % bs): Calculates the exact physical index
        // physical block 42 x 16 slots per block + (20%16) - 42*16*4 = 676
      }

      // boundary of seqs for attention in GPU
      plan.cu_seqlens_q.push_back(plan.cu_seqlens_q.back() + s->n_scheduled);

      plan.cu_seqlens_k.push_back(plan.cu_seqlens_k.back() + end);

      plan.max_seqlen_q = std::max(plan.max_seqlen_q, s->n_scheduled);

      plan.max_seqlen_k = std::max(plan.max_seqlen_k, end);

      plan.context_lens.push_back(end);

      // Add padding to seqs - make same length
      for( size_t b = 0; b < max_blocks; ++b ) {
        plan.block_tables.push_back(
            b < s-> block_table.size() ? s-> block_table[b] : -1
            );
      }
    }

    running_ = batch;
    return plan;
  }

  // Only Full blocks get a content hash. A partial blocks contents still change
  // as tokens arrive so not content-addressable
  void Scheduler::seal_blocks(Sequence& s){

    const int32_t bs = cfg_.block_size;
    int32_t n_full = s.n_cached / bs; // check if any remaining tokens - not full blocks
    uint64_t prev = 0;

      for(int32_t i = 0; i < n_full; ++i){
        if(s.block_hashes[static_cast<size_t>(i)] != 0){ // hash != 0 => block is sealed
          prev = s.block_hashes[static_cast<size_t>(i)]; //  chain blocks
          continue;
        }
        // block not sealed hash=0, chain hash
        prev = block_hash(prev, s.tokens.data() + static_cast<size_t>(i) * bs, bs);

        s.block_hashes[static_cast<size_t>(i)] = prev;

        bm_.register_hash(s.block_table[static_cast<size_t>(i)], prev);
  
    }

  }
  
  /**
   *synchronize the scheduler's internal state with GPU state after generating tokens
   **/
  void Scheduler::commit(const std::vector<int32_t>& next_tokens){

    assert(next_tokens.size() == running_.size());

    for(size_t i = 0; i<running_.size(); ++i){
      Sequence& s = *running_[i];
      s.n_cached += s.n_scheduled;
      s.n_scheduled = 0;

      // n_cached < n_prompt means chunked_prefill in progress.
      // There os no is_chunked flag anywhere - it falls out of this arithematic
      //
      if(s.n_cached < s.n_prompt){  // length of n_cached less than prompt, still prefill left
        seal_blocks(s);
        continue;
      }

      seal_blocks(s); // seal any remaining full blocks from the completing prefill chunk

      s.is_prefill = false;
      s.tokens.push_back(next_tokens[i]);
      s.n_cached = s.n_tokens() - 1; //the new tokens KV not updated 

      bool hit_eos = !s.ignore_eos && next_tokens[i] == s.eos;

      if(hit_eos || s.n_completion() >= s.max_tokens){
        s.state = State::Finished;
        finished_.push_back(s.id);
        release(s);
        ++stats_.total_finished;
      }

    }

    //---------preemption------
    // Newest first: it has computed the least and killing the oldest
    // both wastes the most work and starves whoever has waited the longest.
    
    while(bm_.utilization() > cfg_.max_kv_utilization && running_.size() > 1) {
      size_t victim = running_.size() - 1;
      if(running_[victim]->state == State::Finished) break;
      preempt(victim);
    }

  }

  void Scheduler::preempt(size_t idx) {
    Sequence& s = *running_[idx];
    release(s);
    s.block_table.clear();
    s.block_hashes.clear();
    s.tokens.resize(static_cast<size_t>(s.n_prompt)); // drop generated tokens — redo from the prompt
    s.n_cached = 0;
    s.is_prefill = true;
    s.state = State::Waiting;
    waiting_.push_front(&s);
    running_.erase(running_.begin() + static_cast<long>(idx));
    ++stats_.total_preempted;
  }

  void Scheduler::release(Sequence& s){
    for(int32_t id: s.block_table) bm_.decref(id);
  }

  std::vector<uint64_t> Scheduler::take_finished(){
    std::vector<uint64_t> out;
    out.swap(finished_);
    return out;
  }

  const std::vector<int32_t>& Scheduler::tokens_of(uint64_t id) const {
    return seqs_.at(id) -> tokens;
  }       

  Stats Scheduler::stats() const {
    Stats s = stats_;
    s.waiting = static_cast<int32_t>(waiting_.size());
    s.running = static_cast<int32_t>(running_.size());
    s.kv_util = bm_.utilization();
    s.in_flight_tokens = 0;

    for(const Sequence* q: running_) s.in_flight_tokens += q->n_tokens();
    return s;
  }

}
