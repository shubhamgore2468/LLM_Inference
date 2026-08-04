#pragma once

#include <cstdint>
#include <vector>

namespace kvsched {
  inline uint64_t block_hash(uint64_t prev, const int32_t* tokens, int32_t n) {

    uint64_t h = prev ? prev : 0x9E3779B97F4A7C15ULL;

    for(int32_t i = 0; i<n; ++i) {
      
      h ^= static_cast<uint64_t>(static_cast<uint32_t>(tokens[i]));
      h *= 0x100000001B3ULL;
      h ^= h >> 29;

    }
    return h ? h : 1;
  }  

  inline std::vector<uint64_t> hash_chain(const std::vector<int32_t>& tokens, int32_t block_size){
    std::vector<uint64_t> out;
    int32_t n_full = static_cast<int32_t>(tokens.size()) / block_size;
    out.reserve(static_cast<size_t>(n_full));
    uint64_t h = 0;

    for (int64_t i=0; i<n_full; ++i){
      
      h = block_hash(h, tokens.data() + static_cast<size_t>(i) * block_size, block_size);
      out.push_back(h);

    }
    return out;
  }
}
