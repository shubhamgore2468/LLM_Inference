#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kvsched {

  //Physical block for paged attn

  struct Block {
    int32_t id = -1;
    int32_t ref_count = 0;
    int32_t hash   = 0;
  };

  class BlockManager{
    public:
      BlockManager(int32_t n_blocks, int32_t block_size);

      std::optional<std::vector<int32_t>> allocate(int32_t n);

      void incref(int32_t id);
      void decref(int32_t id);

      void register_hash(int32_t id, uint64_t hash);

      std::optional<int32_t> lookup(uint64_t hash);

      int32_t num_free() const {return static_cast<int32_t>(free_list_.size());}

      int32_t num_total() const {return static_cast<int32_t>(blocks_.size());}
      
      int32_t block_size() const {return block_size_;}

      double utilization() const; 

      const Block& at(int32_t id) const  {return blocks_[id];  }

    private:

      using ListIt = std::list<int32_t>::iterator;

      std::vector<Block> blocks_;
      
      std::list<int32_t> free_list_;

      std::vector<ListIt> pos_;

      std::vector<char> in_free_;

      std::unordered_map<uint64_t, int32_t> cache_;
      int32_t block_size_;

  };

}
