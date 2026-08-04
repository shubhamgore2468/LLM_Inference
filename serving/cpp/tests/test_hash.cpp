// serving/cpp/tests/test_hash.cpp
#include <gtest/gtest.h>

#include "hash.h"

using kvsched::block_hash;
using kvsched::hash_chain;

TEST(Hash, NeverReturnsZero) {
    // 0 is block.h's "no content identity" sentinel.
    for (int32_t t = 0; t < 4096; ++t) {
        int32_t buf[1] = {t};
        EXPECT_NE(block_hash(0, buf, 1), 0u);
    }
}

TEST(Hash, DiffersOnContent) {
    int32_t a[4] = {1, 2, 3, 4};
    int32_t b[4] = {1, 2, 3, 5};
    EXPECT_NE(block_hash(0, a, 4), block_hash(0, b, 4));
}

TEST(Hash, IsOrderSensitive) {
    int32_t a[3] = {7, 8, 9};
    int32_t b[3] = {9, 8, 7};
    EXPECT_NE(block_hash(0, a, 3), block_hash(0, b, 3));
}

// The property that makes it a prefix cache: identical block content under a
// different history must NOT collide.
TEST(Hash, ChainSeparatesIdenticalBlocks) {
    int32_t blk[2] = {42, 42};
    EXPECT_NE(block_hash(0, blk, 2), block_hash(12345, blk, 2));
}

TEST(Hash, SharedPrefixMatchesThenDiverges) {
    std::vector<int32_t> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> b{1, 2, 3, 4, 9, 9, 9, 9};
    auto ha = hash_chain(a, 4);
    auto hb = hash_chain(b, 4);
    ASSERT_EQ(ha.size(), 2u);
    ASSERT_EQ(hb.size(), 2u);
    EXPECT_EQ(ha[0], hb[0]);  // shared first block
    EXPECT_NE(ha[1], hb[1]);
}

TEST(Hash, IgnoresTrailingPartialBlock) {
    std::vector<int32_t> t{1, 2, 3, 4, 5};  // 1 full block of 4, 1 leftover
    EXPECT_EQ(hash_chain(t, 4).size(), 1u);
}
