# WarpKV Architecture: Cuckoo Hashing & Collision Resolution

## 1. Overview
WarpKV utilizes a highly concurrent, GPU-optimized variant of **Cuckoo Hashing** for O(1) average-case lookups. Traditional hash tables resolve collisions using linked lists (chaining) or linear probing, both of which degrade significantly on GPUs due to divergent branching and uncoalesced memory access. Cuckoo hashing ensures a bounded search depth, making it ideal for the SIMT architecture of CUDA.

## 2. Hash Functions and Buckets
Each inserted key is hashed using `xxhash3` to generate a 64-bit fingerprint. This fingerprint is split to determine two distinct, pseudo-random bucket locations within the VRAM hash table:
- **Primary Bucket (`b1`)**
- **Secondary/Alternate Bucket (`b2`)**

Each bucket in WarpKV is designed to hold multiple key-value pairs (typically 8 slots per bucket). This allows multiple threads in a warp to cooperatively read an entire bucket in a single memory transaction, maximizing memory coalescing.

## 3. The Insertion Process
When a warp attempts to insert a batch of keys:
1. It computes `b1` and `b2` for the key.
2. It attempts to atomically insert the key into an empty slot in `b1`.
3. If `b1` is full, it attempts to insert into `b2`.
4. If both `b1` and `b2` are full, a collision occurs, triggering the cuckoo eviction mechanism.

## 4. Cuckoo Eviction (The "Ping-Pong")
To resolve the collision, the inserting thread forcibly evicts an existing key-value pair from the full bucket, taking its place.
1. The evicted "victim" key is then hashed to find its own alternate bucket.
2. If that alternate bucket is also full, the victim evicts *another* key.
3. This creates a chain reaction of evictions (a cuckoo chain) that cascades through the table until an empty slot is eventually found.

## 5. The StashQueue (Fallback Mechanism)
In heavily loaded tables, cuckoo chains can become excessively long or enter infinite cycles (ping-ponging between the same buckets).
- WarpKV enforces a hard limit of `MAX_EVICTION_HOPS` (typically 32).
- If an eviction chain reaches 32 hops without finding an empty slot, the chain is aborted.
- The final victim key is pushed into the **`StashQueue`**—a linear array located in pure VRAM.
- Pushing to the stash is rare (statistically <0.1% of inserts at 50% load factor) but ensures bounded insertion times.

## 6. Lookups
To lookup a key, a warp simply computes `b1` and `b2`. Because a key *must* reside in either `b1`, `b2`, or the `StashQueue`:
1. The warp reads `b1`. If found, it returns.
2. The warp reads `b2`. If found, it returns.
3. If not found in either bucket, it performs a linear scan of the `StashQueue` in VRAM.
Because the stash is kept very small (triggering a background rehash when it exceeds ~4096 elements), this fallback scan is extremely fast (>500 GB/s) and bounded.
