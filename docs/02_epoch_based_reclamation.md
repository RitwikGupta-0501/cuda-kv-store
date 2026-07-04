# WarpKV Architecture: Epoch Based Reclamation (EBR)

## 1. Overview
WarpKV utilizes **Epoch Based Reclamation (EBR)** combined with a double-buffered hash table design. This allows for lock-free, asynchronous background rehashing (resizing and restructuring the table) while concurrent threads continue to insert and look up data without blocking.

## 2. The Double-Buffered Table
At startup, `ArenaAllocator` pre-allocates two complete hash tables in VRAM (`table0` and `table1`). At any given time, only one table is the "active" table.
- **Active Table**: Receives all new inserts and lookups.
- **Standby Table**: Sits empty, waiting for the active table to become full.

## 3. Epochs and Active Readers/Writers
The engine maintains a global atomic `current_epoch` counter and tracks the number of `active_inserts` currently in-flight on the GPU. 
When a host thread submits a batch of operations (inserts or lookups):
1. It acquires the active table pointer and its associated epoch number.
2. For inserts, it increments the global `active_inserts` counter.
3. It launches the CUDA graphs on its designated stream slot.
4. When the stream completes its work, the engine explicitly releases the epoch. If the batch was an insert, it decrements `active_inserts`.

## 4. The Rehashing Process
When the active table approaches its `load_factor_limit` (typically 50% for Cuckoo Hashing to maintain O(1) lookups), or when the `StashQueue` overflows, the CPU triggers a background rehash.

1. **Table Swap**: The engine atomically flips the active table pointer to the standby table and increments the `current_epoch`.
2. **Grace Period**: From this moment on, all *new* incoming operations are routed to the new empty table. However, the background rehash thread *cannot* immediately start migrating data. It must wait for the grace period to end.
3. **Waiting for In-Flight Work**: The rehash thread spins, waiting for all operations that were launched against the old epoch to finish. It monitors the stream events and the `active_inserts` counter until they reach zero for the old epoch.
4. **Data Migration**: Once the old table is guaranteed to have no active writers or readers, the `execute_rehash` CUDA kernel is launched. It scans the old table and the `StashQueue` in parallel, migrating all valid key-value pairs into the new active table using normal cuckoo insertions.
5. **Reclamation**: Once migration is complete, the old table is cleared (zeroed out) using `cudaMemsetAsync`, and it officially becomes the new standby table, ready for the next rehash cycle.

## 5. Benefits
- **Zero Blocking**: User threads submitting inserts or lookups never block waiting for a rehash to complete. They simply acquire whatever the current active table is.
- **High Concurrency**: By double-buffering the multi-gigabyte VRAM allocations, we avoid slow runtime `cudaMalloc` calls during resizing, maintaining predictable high throughput.
