#include "block.h"

#include <cassert>

namespace kvsched {

  BlockManager::BlockManager(int32_t n_blocks, int32_t block_size) : blocks_(static_cast<size_t>(n_blocks)), block_size_(block_size){
    /**
 * @brief Initializes the block manager with a fixed pool of physical blocks.
 *
 * Creates `n_blocks` blocks, assigns each a unique block ID, and places all
 * blocks on the free list. Initially every block has:
 *   - ref_count = 0 (unused)
 *   - hash = 0 (no cached content)
 *
 * @param n_blocks Total number of physical blocks to manage.
 * @param block_size Number of tokens each block can store.
 */
    pos_.resize(static_cast<size_t>(n_blocks));
    in_free_.assign(static_cast<size_t>(n_blocks), 1);

    for(int32_t i=0; i<n_blocks; ++i){
      blocks_[i].id = i;
      free_list_.push_back(i);
      pos_[i] = std::prev(free_list_.end());
    }
  }
/**
 * @brief Allocates a specified number of free blocks.
 *
 * Removes `n` blocks from the free list and marks each as in use by setting
 * its reference count to 1. If a block still contains cached content
 * (identified by its hash), the cache entry is invalidated before reuse,
 * since the block's contents will be overwritten.
 *
 * Allocation is all-or-nothing—if fewer than `n` blocks are available,
 * no blocks are allocated.
 *
 * @param n Number of blocks to allocate.
 *
 * @return std::optional<std::vector<int32_t>>
 *         - Vector of allocated block IDs if successful.
 *         - std::nullopt if `n` is negative or insufficient free blocks exist.
 */  
  std::optional<std::vector<int32_t>> BlockManager::allocate(int32_t n){

    if (n < 0) return std::nullopt;
    if(static_cast<int32_t>(free_list_.size()) < n) return std::nullopt;

    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(n));

    for(int32_t i=0; i<n; ++i){
      
      int32_t id = free_list_.front();
      free_list_.pop_front();
      in_free_[id] = 0;

      Block& b = blocks_[id];

      if(b.hash != 0){
        auto it = cache_.find(b.hash);
        if (it != cache_.end() && it->second == id) cache_.erase(it);
        b.hash = 0;
      }

      assert(b.ref_count == 0);
      b.ref_count = 1;
      out.push_back(id);

    }

    return out;

  }
/**
* @brief Increments the reference count of a block.
*
* If the block is currently free (ref_count == 0) but still cached,
* it is removed from the free list before becoming active again.
* This happens when a previously cached block is reused via a prefix-cache hit.
*
* @param id ID of the block whose reference count should be incremented.
*
* @return void
*/
  void BlockManager::incref(int32_t id){
    Block& b = blocks_[id];
    //Reviving a cached free block (prefix cache hit). Pull it off the free list store
    //it cant be handed ti someone else 
    if(b.ref_count == 0 && in_free_[id]){
      free_list_.erase(pos_[id]);
      in_free_[id] = 0;
    }
    ++b.ref_count;
  }
/**
 * @brief Decrements the reference count of a block.
 *
 * When the reference count reaches zero, the block becomes reclaimable and
 * is added back to the free list. The block's cached hash is intentionally
 * preserved so its contents remain available for future cache lookups until
 * the block is reused.
 *
 * @param id ID of the block whose reference count should be decremented.
 *
 * @return void
 */
  void BlockManager::decref(int32_t id){

    Block& b = blocks_[id];
    assert(b.ref_count > 0);
    --b.ref_count;

    if(b.ref_count == 0){

      free_list_.push_back(id);
      pos_[id] = std::prev(free_list_.end());
      in_free_[id] = 1;

    }

  }
/**
 * @brief Associates a content hash with a block.
 *
 * Registers the block as containing the data identified by `hash`.
 * If another block has already registered the same hash, the registration
 * is ignored ("first writer wins") to avoid duplicate cache entries.
 *
 * @param id ID of the block.
 * @param hash Content hash representing the block's stored data.
 *
 * @return void
 */
  void BlockManager::register_hash(int32_t id, uint64_t hash){
    if(hash == 0) return;
    if(cache_.find(hash) != cache_.end()) return ;
    blocks_[id].hash = hash;
    cache_[hash] = id;
  }

/**
 * @brief Looks up a cached block by its content hash.
 *
 * Searches the content-addressable cache for a block containing the given
 * hash. If found and the block is currently free, its position in the free
 * list is updated to the back, implementing an LRU policy so recently used
 * cached blocks are less likely to be evicted.
 *
 * This function does not increase the block's reference count; the caller
 * must invoke `incref()` if it decides to reuse the block.
 *
 * @param hash Content hash to search for.
 *
 * @return std::optional<int32_t>
 *         - Block ID if a matching cached block exists.
 *         - std::nullopt if no matching block is found.
 */
  std::optional<int32_t> BlockManager::lookup(uint64_t hash){

    auto it = cache_.find(hash);
    if(it == cache_.end()) return std::nullopt;
    int32_t id = it->second;

    if(in_free_[id]){
      free_list_.erase(pos_[id]);

      free_list_.push_back(id);

      pos_[id] = std::prev(free_list_.end());
    }
    return id;
  }
/**
 * @brief Computes the fraction of blocks currently in use.
 *
 * Utilization is defined as:
 *
 *      (# referenced blocks) / (total blocks)
 *
 * or equivalently:
 *
 *      1 - (free blocks / total blocks)
 *
 * Cached blocks on the free list are considered free since they can be
 * immediately reclaimed.
 *
 * @return Fraction of blocks currently allocated, in the range [0.0, 1.0].
 */
  double BlockManager::utilization() const {
    if(blocks_.empty()) return 0.0;
    return 1.0 - static_cast<double>(free_list_.size()) / static_cast<double>(blocks_.size());
  }
}
