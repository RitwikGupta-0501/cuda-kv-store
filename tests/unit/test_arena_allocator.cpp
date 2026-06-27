#include <gtest/gtest.h>

// Phase 2: Arena Allocator Unit Tests

namespace {

class ArenaAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ArenaAllocatorTest, CudaMallocCalledOnce) {
    // TODO: Implement in Phase 2
    // Verify cudaMalloc is called exactly once at startup
    EXPECT_TRUE(true);
}

TEST_F(ArenaAllocatorTest, CorrectBucketCount) {
    // TODO: Implement in Phase 2
    // arena_size / sizeof(Bucket), rounded down to power of 2
    EXPECT_TRUE(true);
}

TEST_F(ArenaAllocatorTest, NoBufferOverrun) {
    // TODO: Implement in Phase 2
    // CUDA memcheck validation
    EXPECT_TRUE(true);
}

TEST_F(ArenaAllocatorTest, PinnedMemoryAllocated) {
    // TODO: Implement in Phase 2
    // Verify cudaMallocHost called for stash
    EXPECT_TRUE(true);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
