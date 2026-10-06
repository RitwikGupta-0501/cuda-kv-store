// =============================================================================
// hash.h — Backward-compatible re-export of warpkv_device.cuh
// =============================================================================
// warpkv_hash32, warpkv_hash32_host, HashPair, and compute_hash_pair have
// been consolidated into warpkv_device.cuh. This header is kept for backward
// compatibility with internal source files.
// New code should include <warpkv/warpkv_device.cuh> directly.
// =============================================================================

#pragma once

#include "../../include/warpkv/warpkv_device.cuh"
