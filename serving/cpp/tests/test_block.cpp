// serving/cpp/tests/test_block.cpp
#include <gtest/gtest.h>

#include "block.h"

using kvsched::BlockManager;

TEST(BlockManager, AllocationIsAllOrNothing) {
    BlockManager bm(4, 16);
    ASSERT_TRUE(bm.allocate(3).has_value());
    EXPECT_EQ(bm.num_free(), 1);

    EXPECT_FALSE(bm.allocate(2).has_value());  // only 1 left
    EXPECT_EQ(bm.num_free(), 1);               // and nothing was consumed
}

TEST(BlockManager, DecrefReturnsToPool) {
    BlockManager bm(4, 16);
    auto ids = bm.allocate(2).value();
    EXPECT_EQ(bm.num_free(), 2);
    for (auto id : ids) bm.decref(id);
    EXPECT_EQ(bm.num_free(), 4);
}

TEST(BlockManager, ForkSharesUntilAllRefsDrop) {
    BlockManager bm(4, 16);
    int32_t id = bm.allocate(1).value()[0];
    bm.incref(id);  // second sequence forks it
    EXPECT_EQ(bm.at(id).ref_count, 2);

    bm.decref(id);
    EXPECT_EQ(bm.num_free(), 3);  // still held by the other sequence
    bm.decref(id);
    EXPECT_EQ(bm.num_free(), 4);
}

// The behavior that separates this from a plain allocator: a block whose refcount
// hit zero is still a valid cache entry.
TEST(BlockManager, CachedBlockSurvivesRefcountZero) {
    BlockManager bm(4, 16);
    int32_t id = bm.allocate(1).value()[0];
    bm.register_hash(id, 0xABCD);
    bm.decref(id);

    auto hit = bm.lookup(0xABCD);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(*hit, id);

    bm.incref(*hit);  // caller claims it
    EXPECT_EQ(bm.num_free(), 3);
}

TEST(BlockManager, ReuseInvalidatesCacheEntry) {
    BlockManager bm(1, 16);
    int32_t id = bm.allocate(1).value()[0];
    bm.register_hash(id, 0xABCD);
    bm.decref(id);

    bm.allocate(1);  // forced to reuse the only block -> eviction happens here
    EXPECT_FALSE(bm.lookup(0xABCD).has_value());
}

TEST(BlockManager, EvictsColdestFirst) {
    BlockManager bm(2, 16);
    auto ids = bm.allocate(2).value();
    bm.register_hash(ids[0], 0x1111);
    bm.register_hash(ids[1], 0x2222);
    bm.decref(ids[0]);  // freed first -> coldest
    bm.decref(ids[1]);

    bm.allocate(1);
    EXPECT_FALSE(bm.lookup(0x1111).has_value());  // coldest evicted
    EXPECT_TRUE(bm.lookup(0x2222).has_value());
}

TEST(BlockManager, LookupBumpsLru) {
    BlockManager bm(2, 16);
    auto ids = bm.allocate(2).value();
    bm.register_hash(ids[0], 0x1111);
    bm.register_hash(ids[1], 0x2222);
    bm.decref(ids[0]);
    bm.decref(ids[1]);

    bm.lookup(0x1111);  // touch the coldest -> it should no longer be coldest
    bm.allocate(1);
    EXPECT_TRUE(bm.lookup(0x1111).has_value());
    EXPECT_FALSE(bm.lookup(0x2222).has_value());
}

TEST(BlockManager, UtilizationTracksReferencedOnly) {
    BlockManager bm(4, 16);
    auto ids = bm.allocate(2).value();
    EXPECT_DOUBLE_EQ(bm.utilization(), 0.5);
    bm.register_hash(ids[0], 0x1111);
    bm.decref(ids[0]);
    // Cached-but-unreferenced counts as free — it's reclaimable under pressure.
    EXPECT_DOUBLE_EQ(bm.utilization(), 0.25);
}
