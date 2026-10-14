# WarpKV — Use Case Tracking & Demo Projects

## Purpose
This document tracks every use case WarpKV intends to serve, the engine requirements each one imposes, and a concrete demo project that proves value to a practitioner in under 5 minutes.

---

## Use Case Overview

| # | Use Case | Wrapper | Market | Priority | Engine Features Required |
|---|---|---|---|---|---|
| 1 | RecSys Dynamic Embeddings | `warpkv.nn.DynamicEmbedding` | Mid-size RecSys teams, researchers | **P0 — Ship first** | Insert-or-get, autograd, slab pool |
| 2 | MoE Expert Caching | `warpkv.ExpertCache` | LLM inference operators | **P1 — Ship second** | LRU eviction, large row stride, prefetch |
| 3 | GNN Dynamic Node Features | `warpkv.nn.DynamicEmbedding` | GNN researchers, fraud/social-graph teams | **P0 — Same wrapper as RecSys** | Insert-or-get, autograd, slab pool |
| 4 | Prefix KV-Block Caching | `warpkv.PrefixCache` | LLM inference engine developers | **P2 — Ship third** | TTL/LRU eviction, large row stride, token hashing utils |

---

## Use Case 1: RecSys Dynamic Embeddings

### The Pain
Recommendation models (DLRM, DCN, DeepFM) map sparse categorical features (user IDs, item IDs, ad IDs, cross-features) to dense embedding vectors. In production:

- Feature vocabularies are **unbounded** — new user IDs appear every second
- The vocabulary is too large to pre-allocate as a single `nn.Embedding(num_embeddings=10B, dim=128)` — that's 4.7 TB of VRAM
- Current workarounds:
  - **Hash trick** (`feature_id % vocab_size`): Loses information through collisions. [Meta's research](https://arxiv.org/abs/1909.02107) shows 1–2% AUC degradation on production models
  - **Pre-built vocabulary + OOV bucket**: Requires a preprocessing pass over the full dataset. New IDs at inference time get lumped into a single OOV vector
  - **TorchRec**: Solves this properly but requires significant infrastructure setup (distributed training, Planner, sharding strategies). Overkill for a single-GPU experiment

### What WarpKV Provides
A drop-in `nn.Module` that acts like `nn.Embedding` but:
- Accepts **any** `int64` ID without a pre-built vocabulary
- Allocates and initializes new embeddings on first access (insert-or-get)
- Lives entirely on GPU — no CPU↔GPU transfer
- Supports `backward()` and works with standard `torch.optim` optimizers
- Resizes transparently when capacity is reached

### Engine Requirements

| Requirement | Detail |
|---|---|
| **Insert-or-get** | Atomic: probe → on miss, claim slab slot + init + insert → return slot index. Must be safe under concurrent threads |
| **Slab pool** | `float32` rows of configurable `embedding_dim` (typically 16–256) |
| **Autograd** | `torch.autograd.Function` with forward (gather from slab) and backward (`atomicAdd` gradients into slab) |
| **Initialization** | Xavier uniform, Kaiming, or zero init for new embeddings. Configurable |
| **Optimizer compatibility** | Slab must be exposable as a parameter-like object that `torch.optim.SparseAdam` or `SGD` can `.step()` on |
| **Resize** | Transparent 2x growth when load factor exceeds threshold. Already implemented via EBR rehash |
| **Eviction** | Not required for training (append-only). Optional LRU for inference serving |
| **Row stride** | Small: 64–1024 bytes (16–256 float32s) |

### Demo Project: Criteo CTR Prediction with Unbounded Features

**What it proves:** WarpKV handles real-world sparse IDs without vocabulary pre-processing, matches or beats hash-trick baselines on prediction quality, and trains at competitive throughput.

**Dataset:** [Criteo Display Ads](https://www.kaggle.com/c/criteo-display-ad-challenge) (public, ~45M samples, 26 categorical features with high cardinality — some features have 10M+ unique values).

**Baseline comparison:**

| Model | Embedding Strategy | Expected AUC | Notes |
|---|---|---|---|
| DeepFM + `nn.Embedding` | Hash trick (`% 1M`) | ~0.792 | Collisions degrade sparse features |
| DeepFM + `nn.Embedding` | Full vocabulary (pre-built) | ~0.803 | Requires preprocessing pass, huge memory |
| DeepFM + `warpkv.nn.DynamicEmbedding` | Insert-on-miss, no hash trick | ~0.803 | **No preprocessing, bounded GPU memory** |

**Script structure:**

```
demos/recsys_criteo/
├── README.md              # Setup instructions, expected results
├── download_data.sh       # Fetches Criteo sample dataset
├── model.py               # DeepFM implementation
├── train_baseline.py      # Training with nn.Embedding + hash trick
├── train_warpkv.py        # Training with warpkv.nn.DynamicEmbedding
├── compare.py             # AUC + throughput comparison, generates table
└── requirements.txt       # torch, scikit-learn, pandas
```

**Key metrics to report:**
- **AUC** (prediction quality — proves no information loss vs. hash trick)
- **Throughput** (samples/sec — proves GPU-native speed)
- **Peak VRAM** (proves bounded memory despite unbounded vocabulary)
- **Lines of code changed** (proves drop-in simplicity — should be <10 lines diff)

**The "aha moment":** Side-by-side diff showing the only change is replacing `nn.Embedding(hashed_vocab_size, dim)` with `warpkv.nn.DynamicEmbedding(dim, device='cuda')` — no vocabulary, no hash trick, no preprocessing — and AUC goes up.

---

## Use Case 2: MoE Expert Caching

### The Pain
Mixture-of-Experts models (Mixtral 8x7B, DeepSeek-V3 256 experts, Qwen-MoE) route each token to a subset of expert FFN blocks. At inference scale:

- A model may have 64–256 experts, each 0.5–2 GB
- Only 2–8 experts activate per token
- Total expert weight: 128–512 GB — far exceeds single-GPU VRAM (24–80 GB)
- Current approaches:
  - **Load all experts (if they fit):** Only works for small MoE models on large GPUs. Doesn't scale
  - **Offload to CPU, load on demand:** PCIe bandwidth (32 GB/s on Gen4) becomes the bottleneck. Loading a 1 GB expert takes ~31ms — unacceptable for real-time inference at tokens/sec rates
  - **Expert parallelism across GPUs:** Requires multi-GPU setup and NCCL communication. Expensive, complex
  - **Ad-hoc caching:** Every inference framework builds its own — none are reusable

### What WarpKV Provides
A GPU-resident expert cache that:
- Maps `expert_id` (int64) to a pre-allocated VRAM slot holding the expert's weight tensor
- Evicts cold experts via LRU/LFU when capacity is reached
- Provides a **prefetch API**: given the router's top-k expert selections for the current batch, asynchronously load predicted experts before they're needed
- Reports cache hit rate for monitoring

### Engine Requirements

| Requirement | Detail |
|---|---|
| **Lookup** | Standard lookup by expert_id → slot index |
| **Insert with eviction** | When cache is full, evict LRU/LFU entry, reclaim its slab slot, insert new expert |
| **Slab pool** | Large row stride: 500 MB – 2 GB per slot. Few slots (32–128), very large payloads |
| **Eviction policy** | LRU (minimum). LFU preferred. Must be O(1) and GPU-friendly |
| **Prefetch** | Async: `cache.prefetch(expert_ids, stream)` — triggers CPU→GPU copy on a background stream before the expert is needed |
| **Autograd** | **Not required** — inference only |
| **Resize** | Not required — fixed capacity based on available VRAM |
| **Hit rate tracking** | Atomic counters for hits/misses, queryable from Python |
| **Row stride** | Very large: 500 MB – 2 GB |

> **Note:** The slab pool design must handle a **bimodal** use pattern: millions of tiny rows (RecSys embeddings) OR dozens of huge rows (expert weights). The simplest approach is to make `row_stride_bytes` a constructor parameter and let the wrapper choose. Internally it's always `base_ptr + slot_index * row_stride`.

### Demo Project: Mixtral Expert Cache Simulator

**What it proves:** GPU-resident expert caching dramatically reduces expert-load latency compared to CPU offloading, with high cache hit rates under realistic routing patterns.

**Setup:** We don't need an actual Mixtral model. We simulate the access pattern:
1. Generate a Zipfian distribution of expert activations (matching empirical MoE routing statistics)
2. For each "request," select top-2 experts from the distribution
3. Compare cache-hit latency (GPU memory access) vs. cache-miss latency (simulated PCIe transfer)

**Baseline comparison:**

| Strategy | Avg expert-load latency | Cache hit rate |
|---|---|---|
| CPU offload (no cache) | ~31ms per expert | 0% |
| WarpKV LRU cache (8 of 64 experts) | < 0.1ms hit / 31ms miss | ~85% (Zipfian) |
| WarpKV LRU cache (16 of 64 experts) | < 0.1ms hit / 31ms miss | ~93% (Zipfian) |

**Script structure:**

```
demos/moe_expert_cache/
├── README.md                # Setup, expected results, how to interpret
├── simulate_routing.py      # Generates Zipfian expert activation traces
├── bench_no_cache.py         # Simulates CPU offload latency (baseline)
├── bench_warpkv_cache.py     # WarpKV LRU expert cache
├── compare.py                # Latency + hit rate comparison, generates charts
├── plot_hitrate_vs_cachesize.py  # Sweep cache size, plot hit rate curve
└── requirements.txt
```

**Key metrics to report:**
- **Average expert-load latency** (ms) — the headline number
- **Cache hit rate** vs. cache size (number of cached experts)
- **P99 latency** — tail latency matters for serving
- **Memory overhead** of the cache metadata (should be negligible vs. expert weights)

**The "aha moment":** A chart showing that caching just 12.5% of experts (8 of 64) cuts average load latency by 85% because of the Zipfian distribution. The cache practically pays for itself.

---

## Use Case 3: GNN Dynamic Node Features

### The Pain
Graph Neural Networks on evolving graphs (fraud detection, social networks, temporal knowledge graphs) must learn representations for nodes/edges that **appear and disappear** over time:

- New users, new transactions, new relationships arrive continuously
- Node feature vocabulary can't be pre-built — you'd have to retrain or re-index constantly
- Current GNN frameworks (PyG, DGL) handle features via:
  - **Static `nn.Embedding`**: Requires knowing all node IDs upfront. New nodes get no learned features
  - **CPU-side feature stores**: `dgl.graphbolt.FeatureStore`, PyG's `FeatureStore` — features live on CPU, transferred per mini-batch via PCIe
  - **Hash-based fallback**: Same collision problems as RecSys

### What WarpKV Provides
The **same `warpkv.nn.DynamicEmbedding`** used for RecSys, but applied to node/edge feature learning:
- New node ID appears in a mini-batch → embedding is auto-initialized and inserted
- Standard GNN message-passing backward pass flows gradients through to the embedding
- No CPU↔GPU transfer — features live on GPU alongside the model

### Engine Requirements

Same as Use Case 1 (RecSys). The `DynamicEmbedding` wrapper is identical — the only difference is how it's used in the training loop.

| Requirement | Same as RecSys? |
|---|---|
| Insert-or-get | ✅ Yes |
| Slab pool | ✅ Yes (same float32 rows) |
| Autograd | ✅ Yes |
| Initialization | ✅ Yes |
| Optimizer compat | ✅ Yes |
| Eviction | ⚠️ Optional but more useful here — graph nodes go stale |
| Row stride | ✅ Same (small, 64–1024 bytes) |

### Demo Project: Temporal Transaction Fraud Detection

**What it proves:** WarpKV handles a continuously growing node set without retraining or re-indexing, outperforms static embeddings on temporal graphs where new entities dominate.

**Dataset:** [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (public, ~200K Bitcoin transaction nodes, 49 timesteps, binary label: licit/illicit). Alternatively, a synthetic temporal graph generator if licensing is an issue.

**Baseline comparison:**

| Model | Node Feature Strategy | F1 (illicit class) | Notes |
|---|---|---|---|
| GCN + static `nn.Embedding` | Pre-built vocab from first 30 timesteps | ~0.72 | New nodes in timesteps 31–49 get OOV embedding |
| GCN + re-indexed `nn.Embedding` | Rebuild vocab every 5 timesteps | ~0.78 | Expensive, loses learned weights on re-index |
| GCN + `warpkv.nn.DynamicEmbedding` | Insert-on-miss, continuous | ~0.80 | **No re-indexing, no OOV, continuous learning** |

**Script structure:**

```
demos/gnn_fraud_detection/
├── README.md                 # Setup, expected results
├── download_data.sh          # Fetches Elliptic dataset
├── graph_builder.py          # Builds temporal graph snapshots
├── model.py                  # 2-layer GCN
├── train_static.py           # Static nn.Embedding baseline
├── train_reindex.py          # Re-indexed nn.Embedding baseline  
├── train_warpkv.py           # warpkv.nn.DynamicEmbedding
├── compare.py                # F1 + throughput comparison
└── requirements.txt          # torch, torch_geometric, pandas
```

**Key metrics to report:**
- **F1 score on illicit class** across timesteps (proves quality on new nodes)
- **Throughput** (nodes/sec during training)
- **Memory** (VRAM usage — should grow gracefully as new nodes appear)
- **Temporal F1 curve** — plot F1 at each timestep, showing WarpKV maintains performance where static baselines degrade

**The "aha moment":** A line chart showing F1 over timesteps. Static embedding F1 drops as new nodes appear (no learned features for them). WarpKV's F1 stays flat because every new node gets an auto-initialized, trainable embedding on first encounter.

---

## Use Case 4: Prefix KV-Block Caching

### The Pain
LLM inference services handle thousands of concurrent requests that often share common prefixes:
- System prompts ("You are a helpful assistant...")
- Few-shot examples (shared across all requests in a task)
- RAG context (same retrieved documents for similar queries)

Without prefix caching, each request recomputes attention KV-states for the shared prefix from scratch. With 1000 requests/sec sharing a 2K-token system prompt on a 70B model, that's enormous redundant compute.

SGLang's RadixAttention and vLLM's automatic prefix caching solve this — but as internal, non-reusable components tightly coupled to their specific attention implementations.

### What WarpKV Provides
A reusable GPU-resident cache of precomputed KV-blocks:
- Key: hash of the token sequence prefix
- Value: the precomputed KV-cache tensors for that prefix (across all layers/heads)
- Eviction: LRU or TTL-based (prefixes go stale when system prompts change)
- Designed to be plugged into any attention implementation, not just one framework

### Engine Requirements

| Requirement | Detail |
|---|---|
| **Lookup** | Hash(token_prefix) → slot index |
| **Insert** | Store computed KV-block at a slab slot |
| **Eviction** | LRU + TTL. Stale prefixes must be evictable |
| **Slab pool** | Very large rows: a prefix KV-block for a 70B model at 2K tokens can be ~2 GB |
| **Autograd** | **Not required** — inference only |
| **Prefix hashing utils** | Helper to compute a rolling hash of token sequences efficiently |
| **Row stride** | Very large: 100 MB – 2+ GB |

### Demo Project: Prefix Cache Hit-Rate Simulator

**What it proves:** A GPU-resident prefix cache achieves high hit rates on realistic multi-tenant serving workloads, translating to significant compute savings.

**Setup:** Simulate a multi-tenant LLM serving workload:
1. Define 10 "tenants" with distinct system prompts (50–500 tokens each)
2. Generate 10K requests with a Zipfian tenant distribution (some tenants are much more active)
3. Each request = system prompt + unique user query
4. Measure cache hit rate and compute savings (tokens not reprocessed)

**Script structure:**

```
demos/prefix_cache/
├── README.md                  # Setup, expected results
├── generate_workload.py       # Multi-tenant request trace generator
├── simulate_no_cache.py       # Baseline: recompute everything
├── simulate_warpkv_cache.py   # WarpKV prefix cache
├── compare.py                 # Hit rate + compute savings
└── requirements.txt
```

**Key metrics to report:**
- **Cache hit rate** vs. number of tenants
- **Compute savings** (% of total prefix tokens avoided via cache hits)
- **Memory cost** of cache vs. compute cost of reprocessing

**The "aha moment":** With 10 tenants and a cache holding just 10 prefix blocks (~20 GB), 92%+ of requests hit the cache, saving ~60% of total prefill compute.

> **Note:** This demo is a **simulator** — it doesn't run an actual LLM. The goal is to prove the caching dynamics. Integration with a real inference engine (vLLM, SGLang) is a v0.3 effort that requires framework-specific attention hooks.

---

## Cross-Cutting Requirements Matrix

This matrix shows which engine features each use case demands. Features needed by P0 use cases are built first.

| Engine Feature | RecSys (P0) | GNN (P0) | MoE (P1) | Prefix (P2) | Build When |
|---|---|---|---|---|---|
| Hash table (int64 → uint32) | ✅ | ✅ | ✅ | ✅ | ✅ Exists |
| Slab pool (configurable stride) | ✅ | ✅ | ✅ | ✅ | **v0.1** |
| Insert-or-get (atomic) | ✅ | ✅ | — | — | **v0.1** |
| Autograd (forward + backward) | ✅ | ✅ | — | — | **v0.1** |
| `nn.Module` wrapper | ✅ | ✅ | — | — | **v0.1** |
| Standard lookup | ✅ | ✅ | ✅ | ✅ | ✅ Exists |
| Delete | — | ⚠️ Optional | — | — | ✅ Exists |
| LRU eviction | — | ⚠️ Optional | ✅ | ✅ | **v0.2** |
| LFU eviction | — | — | ⚠️ Optional | — | v0.3 |
| TTL eviction | — | — | — | ✅ | v0.3 |
| Prefetch API | — | — | ✅ | — | **v0.2** |
| Hit rate counters | — | — | ✅ | ✅ | **v0.2** |
| Large row stride (>1 MB) | — | — | ✅ | ✅ | **v0.2** |
| EBR rehash / resize | ✅ | ✅ | — | — | ✅ Exists |
| Token hashing utils | — | — | — | ✅ | v0.3 |
| Multi-GPU / NCCL | — | — | ⚠️ Optional | — | v0.4+ |

---

## Delivery Plan

| Release | Focus | Demos Shipping | Timeline (1 senior eng) |
|---|---|---|---|
| **v0.1** | Slab pool + autograd + `DynamicEmbedding` | Criteo RecSys, GNN Fraud | 8–10 weeks |
| **v0.2** | LRU eviction + `ExpertCache` | MoE Expert Cache | +4–6 weeks |
| **v0.3** | TTL eviction + `PrefixCache` | Prefix Cache Simulator | +3–4 weeks |

---

## Demo Project Ground Rules

Every demo must follow these rules to maximize impact:

1. **Runs in < 5 minutes** on a single GPU (T4 or better). Long training runs lose attention
2. **Public dataset only** — no proprietary data, no data agreements. Everything downloadable via script
3. **Baseline included** — every demo runs a vanilla PyTorch baseline *and* the WarpKV version, then prints a comparison table. The value must be self-evident without reading a paper
4. **< 10 lines diff** — the WarpKV version should differ from the baseline by as few lines as possible. Drop-in simplicity is the pitch
5. **One script, one number** — `python compare.py` should print the key metric (AUC, F1, latency, hit rate) for both approaches. No manual analysis needed
6. **Reproducible** — pin random seeds, pin package versions, document GPU requirements

---

## Open Questions

- [ ] **Slab pool memory limit strategy:** Should the engine accept a `max_memory_bytes` budget and auto-compute capacity from `row_stride`, or should the user specify `capacity` explicitly? The budget approach is more user-friendly for MoE (user thinks in "I have 20 GB for expert cache"), while explicit capacity is simpler for embeddings
- [ ] **Eviction callback:** When an expert is evicted from the MoE cache, should WarpKV fire a callback so the user can persist it to CPU/disk? Or is that the user's responsibility?
- [ ] **Multi-dtype slab pool:** RecSys uses `float32`, LLM KV-blocks use `float16`/`bfloat16`. Support at engine level (configurable dtype) or wrapper level (reinterpret bytes)?
- [ ] **License:** The repository has GPL-3.0 but the README claims MIT. This must be resolved before any public release. GPL-3.0 is a dealbreaker for many companies
