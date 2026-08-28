# VK Compute — Complete Technical Summary

A Vulkan-based GPU compute engine for running LLM inference (target: **Qwen3.5 9B**, 32 layers, **hybrid quantization** — INT4, INT8, and FP16 layers) on AMD RDNA1-class GPUs (RX 580: 36 CUs, wave64). Three entry points, all gated by `main.exe`:

- **Server mode** (`main.exe`, default): a **persistent** inference daemon. It loads the real safetensors weights once (§4.7), then serves repeated "tokenize → generate" requests over a length-prefixed binary protocol on stdin/stdout (`[uint32 n][n×uint32 ids]` in → `[uint32 m][m×uint32 ids]` out; `n == 0` shuts down). The Python frontend `vk_llm.py` (repo root, run under `.venv` via **uv**) tokenizes text, drives the daemon, and detokenizes output.
- **Validation mode** (`main.exe val`): the original shader harness — randomized test data, weight transpose/quantize/upload, GPU dispatch, comparison against single-threaded CPU references.
- **Memory-info mode** (`main.exe meminfo`): dumps the device memory heaps/types (§3.2) and exits — used to diagnose the VRAM budget (§12).

Both the server and the harness share the same `operation` dispatch core (§3.3).

> **Model status.** The engine runs the **text stack only** (32 transformer layers; the `mtp.*` and `model.visual.*` tensors are ignored). The full-attention and FFN layers are faithful to the reference. The **gated delta-net (linear attention) is currently a simplification**: the shaders implement the sigmoid-`a`/`b` delta rule + `silu(g)` output gate, but the real model's short **`conv1d`** (kernel-4 causal conv over the qkv projection), per-head **`A_log` decay**, **`dt_bias`**, and the q/k **`norm.weight`** head-norm are **loaded into buffers but not yet consumed** (§7.3). As a result the delta-net layers produce a coherent but **not HF-identical** output. See §13 (known issues).

---

## 1. Project Overview

- **Language/stack:** C (harness), GLSL 450 (compute shaders), Vulkan 1.1, glslangValidator, GNU Make (MinGW-w64 / MSYS2 UCRT64), Windows.
- **Two inference phases:**
  - **Decode (token generation):** `GEMV` / split-K shaders — one token (M=1) × weight matrix, pre-compiled into op groups of `DECODE_GROUP = 4` tokens and double-buffered on the queue.
  - **Prefill (first token / prompt processing):** `GEMM2` shaders — prompt tokens × weight matrix, processed in **chunks** of `MODEL_PREFILL_CHUNK = 512` tokens (chunk size = `MODEL_MAX_GEMM`, the state's `maxM`).
- **Precision:** every layer family exists in FP16, INT8 (per-group 256 asymmetric quantization), and INT4 versions. Some shaders (GatedDeltaNet) are precision-agnostic — they operate on already-dequantized float projections.
- **Validation:** every shader has a CPU reference (`*_ref` functions) and a `validate*` function that runs the shader and compares via max absolute error (`main.exe val`).

### Model dimensions used by the harness

| Symbol | Value | Meaning |
|---|---|---|
| `d_model` / `K` | 4096 | hidden size / reduction dimension |
| `N` (FFN) | 12288 | FFN gate+up columns |
| `wo_n` | 4096 | delta-net out projection / residual size |
| `qkv_n` | 10240 | q(16×256) + g(16×256) + k(4×256) + v(4×256); `g` is a sigmoid output gate |
| `heads`, `kv_heads`, `att_dim` | 16 / 4 / 256 | full attention geometry |
| `n_qk`, `n_v`, `dim` | 16 / 32 / 128 | gated delta-net geometry |
| `proj_n` | 12352 | delta-net input projection (Q,K,V,G,A,B) |
| `M` (prefill) | 512 / 16384 | prefill chunk / max chunk (`MODEL_PREFILL_CHUNK` / `MODEL_MAX_GEMM`) |
| `vocab_size` | 81920 (default 248320) | vocab: pruned (81920) or full (248320, padded from 248070 real tokens) — runtime `--vocab-head`/`--vocab-embed` |
| `max_ctx` | 32768 | compile-time `MODEL_MAX_CTX` (buffer/`#define MAXCTX` shader stride); exposed as a runtime limit via `--max-ctx` |
| `max_ops` | 1280 | max `operation`s per op array (`MODEL_MAX_OPS`) |

---

## 2. Repository Structure

```
vk-compute/
├── vk_llm.py                 # Python frontend: start_llm/tokenize/generate (uv venv, drvies main.exe server)
├── tools/tokenize.py  tools/detokenize.py   # standalone tokenize/detokenize helpers
├── Makefile                    # recursive shader build -> bin/shader/*.spv
├── include/                    # C headers
│   ├── buffer.h  data.h  descriptor.h  device.h  dispatch.h
│   ├── fence.h  pipeline.h  session.h  validation.h
│   ├── safetensors.h           # safetensors parser (BF16/F32, 64-bit offsets)
│   ├── model.h                 # model dimensions + layer spec (32 layers, hybrid quant)
│   ├── weights.h               # weight tensors (block-transposed, quantized)
│   ├── state.h                 # activation / KV-cache / scratch buffers
│   └── generate.h              # generator struct + prefill/generateTokens/reset
├── src/
│   ├── main.c                  # arg dispatch: default=server, `val`=harness, `meminfo`
│   ├── compute.c               # serverMain (server loop) + memInfo + model_config spec
│   ├── safetensors.c           # safetensors header parse + BF16→F32 load
│   ├── validation.c            # CPU reference impls + all validate* functions
│   ├── generate.c              # op compiler: chunked prefill, decode groups, lm head
│   ├── weights.c  state.c      # real-weight upload (safetensors→quantize→block-transpose), state buffers
│   ├── session.c device.c buffer.c command.c fence.c   # Vulkan setup
│   ├── descriptor.c pipeline.c dispatch.c              # descriptors, pipelines, dispatch, timing log
│   └── data.c                  # pseudo-random data, fp16/bf16 conversion, transpose_block16, quantize
└── shader/
    ├── Utility/                # shared matmul primitives
    │   ├── FP16/ INT8/ INT4/   # GEMV-*, GEMV-ADD-*, GEMV-SplitK-*, GEMM-*, GEMM-ADD2-*,
    │   │                       # RmsNorm-GEMV-*, LMHead-GEMV-ArgMax-*
    │   ├── RmsNorm-Prologue.comp        # invRms prologue (used by all GEMM2 kernels)
    │   ├── Reduce-GEMV-ADD.comp         # split-K reduce + residual add
    │   └── ArgMax-Reduce.comp           # final argmax over per-workgroup maxes
    ├── Full-Attention/         # QKV projection + RoPE + full attention
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-QKV-* (legacy fused), RmsNorm-QKV-SplitK-*, Reduce-Rope-*
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-QKV-GEMM2-* + Rope-GEMM-* (prefill, split passes)
    │   ├── FP16/ INT8/ INT4/   # Att-full-* (decode), Att-SplitK2-* (decode, split-K)
    │   ├── FP16/ INT8/ INT4/   # Att-QK2-*, Att-PV2-* (prefill, unfused)
    │   ├── Att-Softmax.comp    # prefill softmax pass (fp16 score buffer + smSum reciprocals)
    │   └── Reduce-Att2.comp    # decode split-K attention reduce
    ├── Linear-Attention/       # gated delta-net
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-LinearProj-SplitK-* (decode),
    │   │                       # RmsNorm-LinearProj-GEMM2-* (prefill)
    │   ├── Embed-Gather.comp   # prefill embedding pre-fetch (tied lm-head column gather)
    │   ├── GatedDeltaNet.comp  GatedDeltaNet-GEMM.comp   (precision-agnostic)
    │   └── Reduce-LinearProj.comp       # LinearProj split-K reduce (6-way routing)
    ├── FFN/                    # swiglu feed-forward
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-swiglu-ffn-* (decode GEMV),
    │   │                       # RmsNorm-up-ffn-SplitK-* (decode, flattened gate|up),
    │   │                       # FFN-Down-SplitK-* (decode down projection)
    │   │                       # RmsNorm-swiglu-ffn-GEMM2-* (prefill, dual-matrix),
    │   │                       # RmsNorm-swiglu-flat-GEMM2-* (prefill, flattened)
    │   ├── RmsNorm-swiglu-ffn.comp      (fp32, at root)
    │   └── Swiglu-combine.comp          # silu(gAct) * uAct elementwise pass
    └── Prototype/              # experiments / unused
        # gemv.comp, gemv1..7.comp, gemm.comp, RmsNorm.comp, test.comp,
        # online-softmax.comp, RmsNorm-GEMV.comp, RmsNorm-GEMV-Rope-*.comp,
        # RmsNorm-QKV-score-V.comp_
```

---

## 3. Vulkan Runtime (C side)

### 3.1 Session & device

`createSession()` (src/session.c) creates the Vulkan instance, physical device, logical device, compute queue, command pool, command buffer, fence, and a query pool with `TIMESTAMP_QUERY_COUNT` timestamps for GPU timing.

### 3.2 Buffers — `buffer` (src/buffer.c)

```c
typedef enum { MEMORY_RAM, MEMORY_VRAM } memory_type;  // staging vs device-local
buffer createBuffer(VkDevice, VkPhysicalDevice, const void* data, size_t size, memory_type);
buffer createBufferNamed(VkDevice, VkPhysicalDevice, const void* data, size_t size, memory_type, const char* name);
```

- `MEMORY_RAM` = host-visible staging; `MEMORY_VRAM` = device-local (plus a persistent host-visible staging buffer of equal size for the initial copy).
- `createBufferNamed` stores the 64-byte `buffer.name` label and, via `allocateBufferMemory`, meters per-pool bytes onto two file-scope counters (`device_local` / `host_visible`) — on `vkAllocateMemory` failure it prints `OOM: … for '<name>' (DEVICE_LOCAL|HOST_VISIBLE, N MB) | device_local=… host_visible=…` (see §12).
- `createTransferAndCopy(device, queue, bufs, n)` stages host data into all buffers and copies RAM→VRAM where needed.
- `readBuffer(...)` copies results back to host.

### 3.3 Operation & dispatch — `operation` (include/dispatch.h)

```c
typedef struct operation {
    char shader[128];                 // e.g. "GEMM-FP16.spv"
    buffer buffers[MAX_OP_BUFFERS];   // bound to set=0 bindings 0..n-1
    int bufferCount;
    int pushConstants[MAX_PUSH_CONSTANTS];  // ints, copied verbatim to shader
    int pushConstantCount;
    int dispatchX, dispatchY, dispatchZ;    // vkCmdDispatch dims
} operation;

void execute(session s, operation ops[], int opCount);
```

`execute()` (src/dispatch.c) per op: creates a descriptor set, compiles the shader into a pipeline, binds it, pushes constants, inserts a `VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT` memory barrier **between** ops (so chained ops like LinearProj → GatedDeltaNet → GEMM are correctly ordered), dispatches, and fences. GPU time comes from query-pool timestamps around the whole op chain.

Shader loading (src/pipeline.c) — every op name is prefixed with the output folder:

```c
char shaderPath[160];
snprintf(shaderPath, sizeof(shaderPath), "shader/%s", op->shader);
```

so all `.spv` files live flat in `bin/shader/` regardless of their source folder.

---

## 4. Data & Weight Conventions

### 4.1 Test data (src/data.c)

- `getData(seed, M, N)` — float `[M][N]`, pseudo-random in [-1, 1] from a seed-dependent hash of (i, j).
- `getDataFP16(seed, M, N)` — same values converted to IEEE fp16 via `float_to_fp16`.
- `getDataINT8(seed, M, N)` / `getDataINT4(seed, M, N)` — per-group-of-256 asymmetric quantization: `scale = (max-min)/255` (or `/15` for INT4), `zero = -min`, layout `scale[bj*K + row]`, `bj = col/256`. INT4 packs 2 columns per byte, high nibble first.

### 4.2 Block-transposed weight layout — `transpose_block16`

Weights are stored `[K][N]` on host but uploaded **block-transposed** so each shader thread reads contiguous `uvec4`s. Per output column `n`, 16 consecutive bytes hold a block of k-rows; as `uvec4` the index is:

| Precision | uvec4 index | What one uvec4 contains |
|---|---|---|
| FP16 | `w[(k/8)*N + n]` | 8 halves = 2 vec4 k-rows (`unpackHalf2x16` of `.x,.y` and `.z,.w`) |
| INT8 | `w[(k/16)*N + n]` | 16 bytes = 4 vec4 k-rows (`unpackUint8x4` per component) |
| INT4 | `w[(k/32)*N + n]` | 32 nibbles = 8 vec4 k-rows; a 16-k tile takes low half (`.x,.y`, t even) or high half (`.z,.w`, t odd) |

```c
void transpose_block16(const uint8_t* input, uint8_t* output, int M, int N, int data_type)
```

### 4.3 Quant scale/zero layout

`scale[bj*K + k_float]` and `zero[bj*K + k_float]` with `bj = n/256` (group size 256), stored as `vec4[]`; the vec4 index is `bj*(K/4) + kvec` (`kvec = k/4`). A 16-k tile at k-tile `t` needs `scale[bj*(K/4) + t*4 + kv]` for `kv = 0..3`.

### 4.4 Activations & outputs

- Input x: row-major `[M][K]` float, vec4 index `m*(K/4) + kvec`.
- Outputs: row-major `[M][N]` float.
- KV cache: `[token][row]` — fp16 (`uint16_t` + `packHalf2x16`) for FP16/INT8, or `uint8` + per-(kv-head, token) scale/zero for INT4. Quantized scale/zero live at the **fixed** stride `kvh * MODEL_MAX_CTX + token` in every writer and reader (see gotcha 15).
- RoPE theta: `theta[i] = 1e7^(-i/(dim/2))`, length `dim/2 = 128` (`rope_theta = 1e7`).

### 4.5 Push constants

- Generic GEMM/GEMV/FFN/LinearProj: `{M, N, K}`.
- QKV GEMV: `{M, N, K, kOffset, vOffset}` — position comes from the shared position buffer. Legacy QKV GEMM: `{M, N, K, tokenIdx, gOffset, kOffset, vOffset}` (wrote the new context length to the position buffer — no longer used by the engine). Prefill `RmsNorm-QKV-GEMM2-*`: `{M, N, K}` (raw projection only). `Rope-GEMM-*`: `{N, gOffset, kOffset, vOffset, tokBase}` (tokBase = chunk-absolute first token; the position buffer is **not** touched by prefill shaders — `runPrefill` sets it host-side once via `stateSetPosition` before `finalOps`).
- Attention decode (`Att-full-*` / `Att-SplitK2-*`): no push constants — context_length = position buffer + 1. Attention prefill (`Att-QK2-*` / `Att-Softmax` / `Att-PV2-*`): `{ctxLen, qOff, mRows, headBase}` (ctxLen = chunk-absolute context, qOff = chunk base for the causal limit `qOff + mGlobal`). Legacy `Att-full-GEMM-*`: `{context_length}`.
- GEMM2 prefill kernels: `RmsNorm-swiglu-flat-GEMM2-*` `{M, N, K, off}` (off = FFN gate/up boundary for `upHalf` routing); `RmsNorm-swiglu-ffn-GEMM2-*` / `RmsNorm-LinearProj-GEMM2-*` / `GEMM-ADD2-*` `{M, N, K}`; `RmsNorm-Prologue` `{K}`.
- GatedDeltaNet: `{M}` (GEMM) or `{N_V, N_QK, DIM}` (decode).
- LM head argmax: `{M, N, K}` (M=1, N=vocab) for the GEMV-ArgMax pass; `{vocabSize, doIncrement, passIdx}` for ArgMax-Reduce (doIncrement=1 bumps the position buffer by 1 after a generated token; passIdx tags the write slot for the double-buffered decode groups).
- Embed-Gather: `{V}` (vocab stride of the tied lm head). Embed-RmsNorm-LinearProj GEMV/GEMM: `{M, N, K, V}` (V = vocab size, selects the lm-head column used as the embedding).
- Split-K decode reduce passes: `Reduce-GEMV-ADD` `{N, chunks}` (N = K, chunks = 4); `Reduce-LinearProj` `{N}`; `Reduce-Att2` / `Reduce-Rope` no push constants (N from geometry).

### 4.6 Dispatch geometry

- GEMV shaders: `dispatchX = N/256` (one workgroup of 256 threads per 256 output columns), M=1.
- GEMM shaders: `dispatchX = N/16`, `dispatchY = M/16` (each workgroup computes a 16×16 output tile).
- GEMM2 shaders: `dispatchX = N/TN` (TN=32, or 64 for INT4 GEMM-ADD2 / INT4 QKV-GEMM2), `dispatchY = M/16`; each workgroup computes a 16×TN output tile.
- Flattened swiglu (gate|up in one grid): `dispatchX = 2*N/TN` (N=FFN_N, `upHalf = nBase >= off`), `dispatchY = M/16`.
- Rope-GEMM: `dispatchX = 2*heads + 2*kv_heads` (40 head workgroups: q, g, k, v), `dispatchY = M` (one row per workgroup, 256 threads).
- Embed-Gather: `dispatchX = M`, 256 threads (each row gathers its K floats from the tied lm-head column).
- QKV GEMM: `dispatchX = 24` (heads), `dispatchY = M/16` (head-wide 16×256 tile).
- Attention prefill (QK2): `dispatchX = ctxLen/64` (64-token KV tiles), `dispatchY = (M/16)*4` (m-tiles × head-quads); Softmax: `dispatchX = M`, `dispatchY = 4`; PV2: `dispatchX = 4`, `dispatchY = (M/16)*4`.
- Attention decode: `Att-full` `dispatchX = heads`; `Att-SplitK2` `dispatchX = heads`, `dispatchY = 128` (K-chunks of 4 tiles).
- Split-K decode GEMVs: `dispatchX = N/256`, `dispatchY = 4` (K split over 4 workgroups).
- GatedDeltaNet GEMM: `dispatchX = n_v = 32`, one workgroup per v-head.
- LMHead-GEMV-ArgMax: `dispatchX = (vocab+255)/256`; ArgMax-Reduce: single workgroup.

### 4.7 Real weight loading (src/safetensors.c + src/weights.c)

Weights come from the HuggingFace safetensors shards (`model.safetensors-0000{N}-of-00004.safetensors`) plus, for a pruned vocab, the pre-generated `embed_tokens.<N>.safetensors` / `lm_head.<N>.safetensors`. The pipeline mirrors the synthetic (`getData*`) path but reads real data:

1. `safetensors_open` parses the 8-byte-length JSON header (names/dtype/shape/data_offsets) with a minimal in-C JSON walker; data offsets are 64-bit (`_fseeki64`) — several tensors live past the 2 GB mark in a 5.3 GB shard. Both BF16 and F32 tensors are read; BF16 is widened to float (`bf16_to_float`).
2. Per "logical" tensor, the HF `[out][in]` matrices are concatenated along the output axis and transposed to the engine `[in][out]` layout:
   - full-attn `proj` = `q_proj`(q‖g) ‖ `k_proj` ‖ `v_proj` → 10240 columns (Q‖G‖K‖V);
   - delta `proj` = `in_proj_qkv`(q‖k‖v) ‖ `in_proj_z`(g) ‖ `in_proj_a`(32) ‖ `in_proj_b`(32) → 12352 columns (Q‖K‖V‖G‖A‖B);
   - `gate`/`up`/`down` = `mlp.gate_proj`/`up_proj`/`down_proj`, `out` = `o_proj`/`out_proj`.
3. Quantize per the `spec` `QuantType` (`quantizeDataINT8/INT4` for int8/int4, `float_to_fp16` for fp16), then `transpose_block16`, then upload (`createBufferNamed`). Embeddings and the lm-head (`[K][vocab]`) are built fp16 directly from the BF16 `[vocab][K]` source without a float intermediate.
4. The norm vectors (`input_layernorm`, `post_attention_layernorm`, final `norm`, `q_norm`/`k_norm` [256], delta `norm.weight` [128], `A_log`[32], `dt_bias`[32], `conv1d.weight`[8192×1×4]) are loaded as small float buffers.
5. Results are cached on disk in `bin/weights/<name>_<q>.bin` (the old `tensorLoadFile`/`tensorWriteFile` format) so the expensive quantize+transpose runs once. Cache names are per-layer (`proj_3_INT8`) and vocab-sized (`lmHead_81920_FP16`) — the earlier synthetic code keyed the cache by name only and would have aliased every layer to layer 0.

**Untied embeddings**: `w.embed` (from `embed_tokens`) and `w.lmHead` (from `lm_head`) are separate buffers (`tie_word_embeddings: false`); the old code aliased them. RoPE theta base is `1e7` (`rope_theta`); the partial-RoPE factor (`0.25`) and `rms_norm_eps` (`1e-6`, the engine uses `1e-5`) are **not** yet corrected — see §13.

---

## 5. Build System

`make` recursively finds every `shader/**/*.comp` via a `rwildcard` function and compiles each into a **flat** `bin/shader/<basename>.spv`:

```makefile
rwildcard    = $(foreach d,$(wildcard $(1)*),$(call rwildcard,$(d)/,$(2)) $(filter $(subst *,%,$(2)),$(d)))
SHADERS      := $(call rwildcard,$(SHADER_DIR)/,*.comp)
SHADER_OUT   := $(BIN_DIR)/shader
SHADERS_OBJS := $(addprefix $(SHADER_OUT)/,$(notdir $(SHADERS:.comp=.spv)))

define COMPILE_SHADER
$(SHADER_OUT)/$(notdir $(basename $(1))).spv: $(1)
	@if not exist $(SHADER_OUT_W) mkdir $(SHADER_OUT_W)
	"$(VULKAN_SDK)/Bin/glslangValidator" -V --target-env vulkan1.1 "$(1)" -o $$@
endef
$(foreach f,$(SHADERS),$(eval $(call COMPILE_SHADER,$(f))))
```

Run: `make` builds shaders + `bin/main.exe`; the executable is normally launched by the server/Python path (§11). `make clean` removes `bin/` and `build/` recursively.

> **Windows quirks** (important): the effective recipe shell is `cmd.exe`, so `mkdir`/`if exist` must use **backslashes** (`bin\shader`) — cmd treats `/` as a switch prefix — and folder names must not contain spaces (GNU make word-splits `$(wildcard)`/`$(foreach)` output on spaces, hence `Full-Attention/` and `Linear-Attention/` rather than `Full Attention/`).

---

## 6. Validation Harness

Each `validate*` function in src/validation.c:

1. Generates reference output on CPU (exact float math).
2. Transposes weights with `transpose_block16`, allocates `MEMORY_RAM` input/weight and `MEMORY_VRAM` output buffers, uploads them.
3. Builds an `operation` and runs it.
4. Reads back the GPU output and prints via `report()`:

```c
static void report(const char* name, int idx, const float* out, const float* ref, int count, double ms) {
    float err = 0.0f;
    for (int i = 0; i < count; i++) {
        float e = fabsf(out[i] - ref[i]);
        if (e > err) err = e;
    }
    printf("%s: shader[%d]= %f ref[%d]= %f max_err= %f\n", name, idx, out[idx], idx, ref[idx], err);
    printf("%s time: %.3f ms\n", name, ms);
}
```

Key CPU references:

- `rms_norm_apply` — RMSNorm: `xn = x * gamma / sqrt(mean(x²) + 1e-5)`.
- `gemv_ref_fp32/fp16/int8/int4` and `gemm_ref_fp16/int8/int4` — naive M×K×N loops (int4/8 apply dequant `q*scale - zero`).
- `swiglu_ref_*` — rms-norm → gate & up GEMM → `o = silu(gate) * up`.
- `qkv_rope_ref` — per-head RMSNorm → RoPE with `angle = token * theta[col%128]`.
- `validate_attention` / `validate_attention_multi` — online-softmax attention, causal masked (`t <= m`) in the multi-token version.
- `deltanet_ref` — the sequential gated delta-net recurrence (see §7.3).
- `lmhead_argmax_ref_fp16` — streams one logit column at a time (no full logits array), tracks the running max with smallest-index tie-break.

`validation()` (src/validation.c, reached via `main.exe val`) wires everything: it keeps the original M=1 GEMV validations and adds M=64 (`Mg = 64`) GEMM validations with an M-token input `inputM = getData(4321, Mg, K)`, plus the FP16 lm-head argmax validation over `vocab_size = 81920` (weights `lmHeadFP16 = getDataFP16(15001, K, vocab_size)`, ~640 MB). `src/compute.c` no longer holds the harness — it now hosts the `model_config` layer spec plus `serverMain` (the persistent inference loop) and `memInfo` (device heap dump).

---

## 7. Shader Documentation

Common patterns: `TS = 16` (16×16 tiling), `vec = 4` (vec4 dot products), workgroup = 256 threads (`local_size_x = local_size_y = 16`), shared tiles `Asub[4][16]` / `Bsub[4][16]` (vec4 over k), **k-tile loop bound = `K / TS`** (each iteration advances k by 16 floats; A-tile vec4 index `m*(K/4) + t*4 + kv`).

### 7.1 Utility — matmul primitives

#### `GEMV-*` (decode path, M=1)

Each of 256 threads computes one output column `n`; input row staged in shared as vec4; weight read as `uvec4` per column pair of k-vec4s. FP16 inner loop:

```glsl
#pragma unroll
for (uint k=0;k<dims.K/vec; k+=2) {
    uvec4 rawInput = B[baseRow + globalCol];
    vec4 outVectorA = vec4(unpackHalf2x16(rawInput.x), unpackHalf2x16(rawInput.y));
    vec4 outVectorB = vec4(unpackHalf2x16(rawInput.z), unpackHalf2x16(rawInput.w));
    acc += dot(cache[k], outVectorA);
    acc += dot(cache[k+1], outVectorB);
    baseRow += dims.N;
}
```

INT8 does `k += 4` with `unpackUint8x4` per component × scale − zero; INT4 does `k += 8` with `unpackInt4x4` low/high shifts.

#### `GEMV-ADD-*`

Same as GEMV plus a residual buffer: `C[globalCol] = acc + R[globalCol]` (residual-add for the delta-net output layer).

#### `GEMM-*` (prefill path) — the tiling template

Full source of the foundation shader — every fused GEMM variant derives from this pattern:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define TS 16
#define vec 4

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint M; uint N; uint K; } dims;

layout(set = 0, binding = 0) readonly restrict buffer MatrixA { vec4 A[]; };
layout(set = 0, binding = 1) readonly restrict buffer MatrixB { uvec4 B[]; };
layout(set = 0, binding = 2) writeonly restrict buffer MatrixC { float C[]; };

shared vec4 Asub[4][TS];
shared vec4 Bsub[4][TS];

void main() {
    uint row = gl_LocalInvocationID.y;
    uint col = gl_LocalInvocationID.x;
    uint tid = row * TS + col;
    uint mGlobal = gl_WorkGroupID.y * TS + row;
    uint nGlobal = gl_WorkGroupID.x * TS + col;
    float acc = 0.0;

    for (uint t = 0; t < dims.K / TS; t++) {
        if (tid < 64) {
            uint m = tid / 4;
            uint kv = tid % 4;
            Asub[kv][m] = A[(gl_WorkGroupID.y * TS + m) * (dims.K / vec) + t * 4 + kv];
        }
        if (tid < 32) {
            uint block = tid / TS;
            uint n = tid % TS;
            uvec4 raw = B[(t * 2 + block) * dims.N + (gl_WorkGroupID.x * TS + n)];
            Bsub[block * 2][n]     = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
            Bsub[block * 2 + 1][n] = vec4(unpackHalf2x16(raw.z), unpackHalf2x16(raw.w));
        }
        barrier();
        #pragma unroll
        for (uint kv = 0; kv < 4; kv++) acc += dot(Asub[kv][row], Bsub[kv][col]);
        barrier();
    }
    C[mGlobal * dims.N + nGlobal] = acc;
}
```

**How it works:** workgroup = 16 m-rows × 16 n-cols. Per k-tile of 16 floats: 64 threads stage the A tile (16 rows × 4 vec4s) and 32 threads stage the B tile (FP16: 2 `uvec4` per column = 8 halves each, unpacked into 2 vec4 k-rows); every thread then does 4 `dot()` FMAs (16 k × 1 output element). Loop over `K/16` k-tiles.

INT8 B-load (one `uvec4` per column covers the whole 16-k tile + dequant):

```glsl
if (tid < 16) {
    uint n = tid;
    uint nGlobal = gl_WorkGroupID.x * TS + n;
    uint g = nGlobal / 256;
    uvec4 raw = B[t * dims.N + nGlobal];
    #pragma unroll
    for (uint kv = 0; kv < 4; kv++) {
        vec4 s = scale[g * (dims.K / vec) + t * 4 + kv];
        vec4 z = zeroPoint[g * (dims.K / vec) + t * 4 + kv];
        vec4 q = unpackUint8x4(raw[kv]);
        Bsub[kv][n] = q * s - z;
    }
}
```

INT4 B-load (half of the 32-row `uvec4` selected by k-tile parity):

```glsl
if (tid < 16) {
    uint n = tid;
    uint nGlobal = gl_WorkGroupID.x * TS + n;
    uint g = nGlobal / 256;
    uvec4 raw = B[(t / 2) * dims.N + nGlobal];
    bool hi = ((t & 1) == 1);
    uint c0 = hi ? raw.z : raw.x;
    uint c1 = hi ? raw.w : raw.y;
    Bsub[0][n] = unpackInt4x4(c0, lowShifts) * scale[g*(dims.K/vec)+t*4+0] - zeroPoint[g*(dims.K/vec)+t*4+0];
    Bsub[1][n] = unpackInt4x4(c0, highShifts)* scale[g*(dims.K/vec)+t*4+1] - zeroPoint[g*(dims.K/vec)+t*4+1];
    Bsub[2][n] = unpackInt4x4(c1, lowShifts) * scale[g*(dims.K/vec)+t*4+2] - zeroPoint[g*(dims.K/vec)+t*4+2];
    Bsub[3][n] = unpackInt4x4(c1, highShifts)* scale[g*(dims.K/vec)+t*4+3] - zeroPoint[g*(dims.K/vec)+t*4+3];
}
```

`unpackUint8x4` / `unpackInt4x4`:

```glsl
vec4 unpackUint8x4(uint packedData) {
    uvec4 bytes = (uvec4(packedData) >> uvec4(0, 8, 16, 24)) & uvec4(0xFF);
    return vec4(bytes);
}
const uvec4 lowShifts  = uvec4(0, 4, 8, 12);
const uvec4 highShifts = uvec4(16, 20, 24, 28);
vec4 unpackInt4x4(uint packedComponent, uvec4 shiftOffsets) {
    return vec4(uvec4(packedComponent) >> shiftOffsets & uvec4(0xF));
}
```

#### `GEMM-ADD-*`

Generic GEMM + residual: `C[mGlobal*dims.N + nGlobal] = acc + R[mGlobal*dims.N + nGlobal]`.

#### `RmsNorm-GEMV-*`

Fused RMSNorm + GEMV for decode: stage `x * gamma` into shared while computing `sum(x²)` via subgroup add → `inv_rms`, then the GEMV loop with `cache[k] * inv_rms`.

#### `LMHead-GEMV-ArgMax-FP16` (decode, output token selection)

FP16 GEMV over the lm-head weight `[K][vocab]` fused with a workgroup-wide argmax reduction — one dispatch pass, no full logits buffer (the largest intermediate is 320 × 4 B per output buffer). Workgroup = 256 threads, one output column each, `dispatchX = (vocab+255)/256`. Bindings: `0 = x (vec4[])`, `1 = w (uvec4[])`, `2 = maxValue (float[], fp32)`, `3 = maxIndex (uint[])`; push constants `{M, N, K}`. Full source:

```glsl
#version 450
#define TS 256
#define vec 4
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = TS, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint M; uint N; uint K; } dims;

layout(set = 0, binding = 0) readonly restrict buffer MatrixA { vec4 A[]; };
layout(set = 0, binding = 1) readonly restrict buffer MatrixB { uvec4 B[]; };
layout(set = 0, binding = 2) writeonly restrict buffer MaxValueBuffer { float maxValue[]; };
layout(set = 0, binding = 3) writeonly restrict buffer MaxIndexBuffer { uint maxIndex[]; };

shared vec4 Asub[2][TS];
shared float redValue[TS];
shared uint redIndex[TS];

void main() {
    uint col = gl_LocalInvocationID.x;
    uint globalCol = gl_WorkGroupID.x*TS + col;
    uint numTiles = (dims.K + TS - 1) / TS / vec;

    float acc = 0.0;
    if (globalCol < dims.N) {
        Asub[0][col] = A[col];
    }
    barrier();
    uint baseRow = 0;
    for (uint t = 0; t < numTiles; t++) {
        uint cur = t & 1, nxt = (t+1) & 1;
        if (globalCol < dims.N) {
            if (t+1 < numTiles) {
                Asub[nxt][col] = A[(t+1)*TS + col];
            }

            #pragma unroll
            for (uint k=0;k<TS; k+=2) {
                uvec4 rawInput = B[baseRow + globalCol];
                vec4 outVectorA = vec4(unpackHalf2x16(rawInput.x), unpackHalf2x16(rawInput.y));
                vec4 outVectorB = vec4(unpackHalf2x16(rawInput.z), unpackHalf2x16(rawInput.w));

                acc += dot(Asub[cur][k], outVectorA);
                acc += dot(Asub[cur][k+1], outVectorB);
                baseRow += dims.N;
            }
        }
        barrier();
    }
    if (globalCol >= dims.N) {
        acc = uintBitsToFloat(0xFF800000u);
    }

    redValue[col] = acc;
    redIndex[col] = globalCol;
    barrier();
    for (uint stride = TS/2; stride > 0; stride >>= 1) {
        if (col < stride) {
            float otherValue = redValue[col + stride];
            uint otherIndex = redIndex[col + stride];
            if (otherValue > redValue[col] || (otherValue == redValue[col] && otherIndex < redIndex[col])) {
                redValue[col] = otherValue;
                redIndex[col] = otherIndex;
            }
        }
        barrier();
    }
    if (col == 0) {
        maxValue[gl_WorkGroupID.x] = redValue[0];
        maxIndex[gl_WorkGroupID.x] = redIndex[0];
    }
}
```

**How it works:** the GEMV loop is identical to `GEMV-FP16` (ping-pong `Asub`, 2 vec4 k-rows per `uvec4` weight load). Out-of-range threads (`globalCol >= N`, partial last workgroup) skip the compute but still execute every barrier, and carry `-inf` (`uintBitsToFloat(0xFF800000u)`) so they can never win the reduction. Then a shared tree reduction over 256 threads finds the max logit and its column; ties resolve to the **smaller index** for determinism. Thread 0 writes the workgroup's winner to `maxValue[wgID]` / `maxIndex[wgID]` (fp32 value + global column index).

#### `ArgMax-Reduce.comp` (second stage, precision-agnostic)

One workgroup of 256 threads reduces the per-workgroup winners from the first pass to a single token id. Bindings: `0 = maxValue (float[])`, `1 = maxIndex (uint[])`, `2 = result (uint[])`, `3 = position (uint[])`; push constant `{vocabSize, doIncrement}`; `dispatchX = 1`. Full source:

```glsl
#version 450
#define TS 256

layout(local_size_x = TS, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Params {
    uint vocabSize;
    uint doIncrement;
} params;

layout(set = 0, binding = 0) readonly restrict buffer MaxValueBuffer { float maxValue[]; };
layout(set = 0, binding = 1) readonly restrict buffer MaxIndexBuffer { uint maxIndex[]; };
layout(set = 0, binding = 2) writeonly restrict buffer ResultBuffer { uint result[]; };
layout(set = 0, binding = 3) buffer PositionBuffer { uint position[]; };

shared float redValue[TS];
shared uint redIndex[TS];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint numGroups = (params.vocabSize + TS - 1) / TS;

    float bestValue = uintBitsToFloat(0xFF800000u);
    uint bestIndex = 0;

    for (uint i = tid; i < numGroups; i += TS) {
        float v = maxValue[i];
        uint idx = maxIndex[i];
        if (v > bestValue || (v == bestValue && idx < bestIndex)) {
            bestValue = v;
            bestIndex = idx;
        }
    }

    redValue[tid] = bestValue;
    redIndex[tid] = bestIndex;
    barrier();
    for (uint stride = TS/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float otherValue = redValue[tid + stride];
            uint otherIndex = redIndex[tid + stride];
            if (otherValue > redValue[tid] || (otherValue == redValue[tid] && otherIndex < redIndex[tid])) {
                redValue[tid] = otherValue;
                redIndex[tid] = otherIndex;
            }
        }
        barrier();
    }
    if (tid == 0) {
        result[0] = redIndex[0];
        if (params.doIncrement == 1u) position[0] = position[0] + 1u;
    }
}
```

**How it works:** `numGroups = ceil(vocabSize/256)` (= 320 for vocab 81920); each thread strided-loads `maxValue[i]`/`maxIndex[i]` in chunks of 256 (`i += TS` loop), keeping the running best with the same smallest-index tie-break, then one shared tree reduction; thread 0 writes the final token id to `result[0]` and, when `doIncrement == 1` (decode), increments the shared position buffer by exactly one — the single per-token position update, kept out of any per-layer shader so N layers can never bump it N×. Both shaders are chained as two `operation`s in one `execute()` call (the dispatch harness inserts a write→read barrier between ops).

---

### 7.2 Full Attention

#### `RmsNorm-QKV-*` (decode, one token)

Workgroup = one head (256 columns). RMSNorm over K → project `N = 10240` columns (q 0..4095, g 4096..8191, k 8192..9215, v 9216..10239); `g` is the sigmoid output gate → for q/k heads: per-head RMSNorm over 256 cols, then **RoPE**: `angle = pos * theta[col%128]`, first half `a·cos − b·sin`, second half `a·sin + b·cos` (a/b from opposite halves) → q written to `qOut`, k/v stored to the KV cache at slot `pos * (vOffset − kOffset)`. INT4 additionally quantizes k/v per (head, token) with min/max over the 256 cols and writes scale/zero. **Position source:** the write slot and RoPE index come from a shared `uint[1]` position buffer (last binding) instead of push constants — the value is the current context length (how many tokens are already cached).

#### `RmsNorm-QKV-GEMM-*` (prefill, head-wide tile — **legacy, unwired**)

Superseded by the `RmsNorm-QKV-GEMM2-*` + `Rope-GEMM-*` pair (§7.6); kept compiled for the validation harness (`validateQkvRopeGEMM*`). Workgroup = (head, 16-token block); output tile = 16 tokens × 256 columns; the 16×16 k-tiling from §7.1 inside. The k/v cache slot is `(tokenIdx + mGlobal) * cacheRows` and RoPE uses `tokenIdx + mGlobal` (`tokenIdx` = chunk base, kept as a push constant so layers in the same chunk stay consistent). After the projections, the first workgroup's thread 0 writes the new context length to the shared position buffer: `position[0] = tokenIdx + M` — an **absolute** write, so every layer writing the same value is idempotent (last-write-wins is harmless; `execute()` barriers serialize it). Full FP16 source:

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define TS 16
#define vec 4
#define HEAD_DIM 256
#define HALF 128

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions {
    uint M; uint N; uint K; uint tokenIdx; uint kOffset; uint vOffset; uint context_length;
} param;

const float EPSILON = 1e-5;

layout(set = 0, binding = 0) buffer DataBuffer { vec4 x[]; };
layout(set = 0, binding = 1) readonly buffer GammaBuffer { vec4 gamma[]; };
layout(set = 0, binding = 2) readonly buffer weightBuffer { uvec4 w[]; };
layout(set = 0, binding = 3) readonly buffer thetaBuffer { float theta[]; };
layout(set = 0, binding = 4) buffer qOutBuffer { float qOut[]; };
layout(set = 0, binding = 5) buffer kCacheBuffer { uint16_t kCache[]; };
layout(set = 0, binding = 6) buffer vCacheBuffer { uint16_t vCache[]; };

shared vec4 Asub[4][TS];
shared vec4 Bsub[4][HEAD_DIM];
shared float row_sums[TS][TS];
shared float inv_rms_sh[TS];
shared float acc_sh[TS][HEAD_DIM];
shared float norm_sh[TS][HEAD_DIM];
shared float inv_rms2_sh[TS];

void main() {
    uint row = gl_LocalInvocationID.y;
    uint col = gl_LocalInvocationID.x;
    uint tid = row * TS + col;
    uint head = gl_WorkGroupID.x;
    uint mGlobal = gl_WorkGroupID.y * TS + row;
    uint nBase = head * HEAD_DIM;
    uint cacheRows = param.vOffset - param.kOffset;

    // per-token RMSNorm over K: 16 threads per row, tree reduction
    float sum_sq = 0.0;
    for (uint i = 0; i < param.K / (TS * vec); i++) {
        vec4 v = x[mGlobal * (param.K / vec) + col * (param.K / (TS * vec)) + i];
        sum_sq += dot(v, v);
    }
    row_sums[row][col] = sum_sq;
    barrier();
    for (uint stride = TS / 2; stride > 0; stride >>= 1) {
        if (col < stride) row_sums[row][col] += row_sums[row][col + stride];
        barrier();
    }
    if (col == 0) inv_rms_sh[row] = inversesqrt(row_sums[row][0] / float(param.K) + EPSILON);
    barrier();

    // GEMM: 16 tokens x 256 cols, k-tiles of 16
    float acc[16];
    #pragma unroll
    for (uint j = 0; j < 16; j++) acc[j] = 0.0;

    for (uint t = 0; t < param.K / TS; t++) {
        if (tid < 64) {
            uint m = tid / 4;
            uint kv = tid % 4;
            uint mg = gl_WorkGroupID.y * TS + m;
            Asub[kv][m] = x[mg * (param.K / vec) + t * 4 + kv] * gamma[t * 4 + kv] * inv_rms_sh[m];
        }
        for (uint i = tid; i < HEAD_DIM * 2; i += TS * TS) {
            uint block = i / HEAD_DIM;
            uint n = i % HEAD_DIM;
            uvec4 raw = w[(t * 2 + block) * param.N + (nBase + n)];
            Bsub[block * 2][n]     = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
            Bsub[block * 2 + 1][n] = vec4(unpackHalf2x16(raw.z), unpackHalf2x16(raw.w));
        }
        barrier();
        #pragma unroll
        for (uint kv = 0; kv < 4; kv++) {
            vec4 a = Asub[kv][row];
            #pragma unroll
            for (uint j = 0; j < 16; j++) acc[j] += dot(a, Bsub[kv][col * 16 + j]);
        }
        barrier();
    }

    // stage accumulators into shared (head-wide access for QK-norm / RoPE)
    #pragma unroll
    for (uint j = 0; j < 16; j++) acc_sh[row][col * 16 + j] = acc[j];
    barrier();

    // v heads: store raw to cache (no norm, no rope)
    if (nBase >= param.vOffset) {
        uint kvHead = head - param.vOffset / HEAD_DIM;
        #pragma unroll
        for (uint j = 0; j < 16; j++) {
            uint d = col * 16 + j;
            vCache[mGlobal * cacheRows + kvHead * HEAD_DIM + d] = uint16_t(packHalf2x16(vec2(acc_sh[row][d])));
        }
        return;
    }

    // q/k heads: per-head RMSNorm over 256 cols
    float ss = 0.0;
    #pragma unroll
    for (uint j = 0; j < 16; j++) {
        float v2 = acc_sh[row][col * 16 + j];
        ss += v2 * v2;
    }
    row_sums[row][col] = ss;
    barrier();
    for (uint stride = TS / 2; stride > 0; stride >>= 1) {
        if (col < stride) row_sums[row][col] += row_sums[row][col + stride];
        barrier();
    }
    if (col == 0) inv_rms2_sh[row] = inversesqrt(row_sums[row][0] / float(HEAD_DIM) + EPSILON);
    barrier();

    // RoPE (angle uses the token's absolute index)
    float token = float(param.tokenIdx + mGlobal);
    #pragma unroll
    for (uint j = 0; j < 16; j++) {
        uint d = col * 16 + j;
        float angle = token * theta[d & (HALF - 1u)];
        float cs = cos(angle);
        float sn = sin(angle);
        float v;
        if (d < HALF)
            v = acc_sh[row][d] * cs - acc_sh[row][d + HALF] * sn;
        else
            v = acc_sh[row][d - HALF] * sn + acc_sh[row][d] * cs;
        norm_sh[row][d] = v * inv_rms2_sh[row];
    }
    barrier();

    if (nBase < param.kOffset) {
        #pragma unroll
        for (uint j = 0; j < 16; j++) {
            uint d = col * 16 + j;
            qOut[mGlobal * param.kOffset + nBase + d] = norm_sh[row][d];
        }
    } else {
        uint kvHead = head - param.kOffset / HEAD_DIM;
        #pragma unroll
        for (uint j = 0; j < 16; j++) {
            uint d = col * 16 + j;
            kCache[mGlobal * cacheRows + kvHead * HEAD_DIM + d] = uint16_t(packHalf2x16(vec2(norm_sh[row][d])));
        }
    }
}
```

INT8 variant: B-load is 256 `uvec4`s (one per column, full 16-k tile) + scale/zero vec4s; cache stays fp16. INT4 variant: parity-selected nibble half (like §7.1), `uint8` k/v cache plus per-(kv-head, token) min/max quantization (`quantizeParams`/`quantize` helpers with per-row `qscale_sh[TS]`/`qzero_sh[TS]`).

#### `Att-full-*` (decode, one query)

Workgroup = one query head. Query staged in shared (256 floats). Online softmax over KV-token tiles of 256: per tile compute scores `q·k` (fp16 or dequantized int8/int4 keys), tile max via tree reduction, exp, tile sum, then rescale running accumulator with `alpha = exp(m_prev − m_next)` / `beta = exp(tile_max − m_next)` and accumulate `P·V` per output dim. Output `o = acc / l`. Context length comes from the shared position buffer (`context = position[0] + 1`) — no push constants — so the decode chain is fully GPU-driven; the decode QKV shader of the same layer writes the new token's k/v at slot `position[0]` first, and the +1 makes the total context correct for attention.

#### `Att-full-GEMM-*` (prefill, flash attention, causal)

Workgroup = (head, q-tile); 16 queries × all KV tokens; 16×16 score tiles; causal mask `kvTok <= mTok`. Full FP16 source:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define TS 16
#define HEAD_DIM 256
#define KV_HEADS 4
#define KV_TOTAL_ROWS (KV_HEADS * HEAD_DIM)
#define Q_DIM (16 * HEAD_DIM)
#define NEG_INF -1e30

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint context_length; } param;

layout(set = 0, binding = 0) readonly buffer keyBuffer   { uint16_t key[]; };
layout(set = 0, binding = 1) readonly buffer valueBuffer { uint16_t value[]; };
layout(set = 0, binding = 2) readonly buffer queryBuffer { vec4 query[]; };
layout(set = 0, binding = 3) writeonly buffer outBuffer  { float y[]; };

shared vec4    Qs[HEAD_DIM / 4][TS];   // 16 KB
shared f16vec4 KVs[HEAD_DIM / 4][TS]; // 8 KB, reused for K then V
shared float s_max[TS][TS];
shared float s_exp[TS][TS];
shared float s_p[TS][TS];

vec4 loadHalf4(uint b0, uint b1, uint b2, uint b3) {
    return vec4(unpackHalf2x16(b0 | (b1 << 16)),
                unpackHalf2x16(b2 | (b3 << 16)));
}

void main() {
    uint r = gl_LocalInvocationID.y;
    uint c = gl_LocalInvocationID.x;
    uint tid = r * TS + c;
    uint h = gl_WorkGroupID.x;
    uint kvh = h / KV_HEADS;
    uint mBase = gl_WorkGroupID.y * TS;
    uint mTok = mBase + r;

    // stage Q tile (16 tokens x 256 dims)
    for (uint i = tid; i < (HEAD_DIM / 4) * TS; i += TS * TS) {
        uint dv = i / TS;
        uint tok = i % TS;
        Qs[dv][tok] = query[(mBase + tok) * 1024 + h * (HEAD_DIM / 4) + dv];
    }
    barrier();

    float m_prev = NEG_INF;
    float l_prev = 0.0;
    vec4 oacc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) oacc[j] = vec4(0.0);

    uint numTiles = (param.context_length + TS - 1) / TS;
    for (uint t = 0; t < numTiles; t++) {
        uint kvTok = t * TS + c;

        // stage K tile from KV cache ([token][row] layout)
        for (uint i = tid; i < (HEAD_DIM / 4) * TS; i += TS * TS) {
            uint dv = i / TS;
            uint tok = i % TS;
            uint base = (t * TS + tok) * KV_TOTAL_ROWS + kvh * HEAD_DIM + dv * 4;
            KVs[dv][tok] = f16vec4(loadHalf4(uint(key[base]), uint(key[base + 1]), uint(key[base + 2]), uint(key[base + 3])));
        }
        barrier();

        // scores with causal mask (64 vec4 dots over the 256-dim head)
        float score = NEG_INF;
        if (kvTok < param.context_length && kvTok <= mTok) {
            score = 0.0;
            for (uint dv = 0; dv < HEAD_DIM / 4; dv++)
                score += dot(Qs[dv][r], vec4(KVs[dv][c]));
        }
        s_max[r][c] = score;
        barrier();
        for (uint stride = TS / 2; stride > 0; stride >>= 1) {
            if (c < stride) s_max[r][c] = max(s_max[r][c], s_max[r][c + stride]);
            barrier();
        }
        float tile_max = s_max[r][0];
        float m_next = max(m_prev, tile_max);
        float alpha = (m_prev == NEG_INF) ? 0.0 : exp(m_prev - m_next);
        float beta = exp(tile_max - m_next);
        l_prev = alpha * l_prev;

        // exp(score - tile_max) kept in s_p; sum reduced in s_exp
        float e = (score == NEG_INF) ? 0.0 : exp(score - tile_max);
        s_p[r][c] = e;
        s_exp[r][c] = e;
        barrier();
        for (uint stride = TS / 2; stride > 0; stride >>= 1) {
            if (c < stride) s_exp[r][c] += s_exp[r][c + stride];
            barrier();
        }
        l_prev += beta * s_exp[r][0];
        #pragma unroll
        for (uint j = 0; j < 4; j++) oacc[j] *= alpha;

        // stage V tile (reuses the same shared array as K)
        for (uint i = tid; i < (HEAD_DIM / 4) * TS; i += TS * TS) {
            uint dv = i / TS;
            uint tok = i % TS;
            uint base = (t * TS + tok) * KV_TOTAL_ROWS + kvh * HEAD_DIM + dv * 4;
            KVs[dv][tok] = f16vec4(loadHalf4(uint(value[base]), uint(value[base + 1]), uint(value[base + 2]), uint(value[base + 3])));
        }
        barrier();

        // P * V: thread owns 16 output dims (4 vec4 accumulators)
        for (uint kk = 0; kk < TS; kk++) {
            float p = s_p[r][kk] * beta;
            #pragma unroll
            for (uint j = 0; j < 4; j++) oacc[j] += p * vec4(KVs[c * 4 + j][kk]);
        }
        m_prev = m_next;
        barrier();
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint base = mTok * Q_DIM + h * HEAD_DIM + c * 16 + j * 4;
        vec4 o = (l_prev > 0.0) ? (oacc[j] / l_prev) : vec4(0.0);
        y[base + 0] = o.x;
        y[base + 1] = o.y;
        y[base + 2] = o.z;
        y[base + 3] = o.w;
    }
}
```

**How it works:** Q staged once per workgroup. Per 16-KV-token tile: K staged (fp16), scores = 64 vec4 dots/thread with causal mask, row-wise tile max via tree reduction, online-softmax rescale (`alpha`/`beta` per flash-attention), `exp(score − tile_max)` kept in `s_p` for the P·V step (the sum reduction overwrites `s_exp` in place). V then staged into the same shared buffer and multiplied with the per-row probabilities. Output `O = acc / l`. INT8 is identical (cache is fp16); INT4 dequantizes `uint8` cache with per-(kv-head, token) scale/zero during K/V staging.

---

### 7.3 Linear Attention (Gated DeltaNet)

#### `RmsNorm-LinearProj-*`

RMSNorm + single GEMM over `N = 12352`, output routed by column with per-token strides: `q[2048]` (cols < 2048), `k[2048]` (< 4096), `v[4096]` (< 8192), `g[4096]` (< 12288), `a[32]` (< 12320), `b[32]` (else).

#### `Embed-RmsNorm-LinearProj-*` (token-embedding fused variant, FP16 only)

Replaces the `x` input with a token-id lookup: the token id(s) (a `uint` buffer, GEMV: 1 element, GEMM: `M` elements) index into the **tied lm-head buffer** — the same block-transposed FP16 `uvec4[]` layout as `LMHead-GEMV-ArgMax-FP16.comp` binding 1 — to fetch the 4096-dim embedding row, then RMSNorm + LinearProj proceed exactly as `RmsNorm-LinearProj-*`. This shader is meant to run right after the LM-head argmax (token selection) so the next decode step can start from the selected token's embedding without CPU readback.

- **Bindings (GEMV and GEMM):** `0 = tokenIds (uint[])`, `1 = lm (uvec4[], transposed lm head [K][vocab])`, `2 = gamma (vec4[])`, `3 = w (uvec4[], transposed w_in [K][12352])`, `4..9 = q/k/v/g/a/b out (float[])`. Push constants `{M, N, K, V}`.
- **Embedding fetch:** vec4 index `i` (0..1023) reads `lm[(i/2)*V + tok]`; even `i` takes `unpackHalf2x16(raw.x), unpackHalf2x16(raw.y)`, odd `i` takes `.z,.w` (each `uvec4` = 2 vec4 k-rows of a column).
- **GEMV (`Embed-RmsNorm-LinearProj-FP16.comp`):** 256 threads/col, subgroup RMSNorm sum identical to `RmsNorm-LinearProj-FP16`; `dispatchX = (12352+255)/256 = 49`.
- **GEMM (`Embed-RmsNorm-LinearProj-GEMM-FP16.comp`):** 16×16 tiled clone of `RmsNorm-LinearProj-GEMM-FP16`; per-token embedding fetched by `tokenIds[mGlobal]` in both the sum-sq pass and the A-tile staging; `dispatchX = 12352/16 = 772`, `dispatchY = M/16`.

Note the lm-head column fetch is strided by `V` uvec4s (column-major access into a 640 MB buffer), so the GEMM variant is slower than the plain `x`-input GEMM (~150 ms vs ~45 ms at M=64) — a layout artifact of tied embeddings, not a correctness issue.

#### `GatedDeltaNet.comp` (decode, one token)

Workgroup = one v-head; state `S` is a 128×128 matrix per head in VRAM. Per token:

- `alpha = 1/(1+exp(a))`, `beta = 1/(1+exp(-b))` — **per-token and per value-head** (`aRaw[h]`/`bRaw[h]`, 32 values; the earlier fix changed these from `aRaw[qk]`/`bRaw[qk]`, 16 values, to match the real `in_proj_a/b` shape `[32, 4096]`)
- `delta = V − S·K`
- `y = alpha·(S·Q) + beta·delta·(K·Q)`
- `yGated = y · silu(G)`
- state update: `S ← alpha·S + beta·delta·Kᵀ`

> **Not implemented (current gap).** The shader above is the *simplified* gated delta rule. The real Qwen3.5 linear-attention block also has a short causal **`conv1d`** (kernel 4, depthwise over the 8192-dim qkv projection), per-head **`A_log`** exponential decay, **`dt_bias`**, and a q/k head **`norm.weight`** (128-dim), all of which are loaded into `w.conv`/`w.aLog`/`w.dtBias`/`w.attnNorm` (§4.7) but **not yet consumed by `GatedDeltaNet.comp`/`GatedDeltaNet-GEMM.comp`**. Until these are folded in (plus `rms_norm_eps = 1e-6` and the partial RoPE `0.25` correction in §13), the delta-net layers are coherent but not HF-identical.

#### `GatedDeltaNet-GEMM.comp` (prefill, chunked linear attention)

Workgroup = one v-head; processes all M tokens in chunks of 16 inside the shader (state persists across chunks in VRAM, synchronized by `barrier()`). Full source:

```glsl
#version 450
#define TS 16
#define DIM 128

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint M; } param;

layout(set = 0, binding = 0) readonly buffer QBuffer { float Q[]; };
layout(set = 0, binding = 1) readonly buffer KBuffer { float K[]; };
layout(set = 0, binding = 2) readonly buffer VBuffer { float V[]; };
layout(set = 0, binding = 3) readonly buffer GBuffer { float G[]; };
layout(set = 0, binding = 4) readonly buffer ARawBuffer { float aRaw[]; };
layout(set = 0, binding = 5) readonly buffer BRawBuffer { float bRaw[]; };
layout(set = 0, binding = 6) buffer StateBuffer { float S[]; };
layout(set = 0, binding = 7) buffer YGatedBuffer { float yGated[]; };

shared float Ks[TS][DIM];
shared float Dk[TS][TS];
shared float Dq[TS][TS];
shared float delta[TS][DIM];
shared float alpha_sh[TS];
shared float beta_sh[TS];
shared float alpha_P[TS + 1];
shared float w_sh[TS];
shared float skv[DIM];
shared float sqv[DIM];

void main() {
    uint r = gl_LocalInvocationID.y;
    uint c = gl_LocalInvocationID.x;
    uint tid = r * TS + c;
    uint h = gl_WorkGroupID.x;
    uint qk = h >> 1;

    uint numChunks = param.M / TS;

    for (uint ch = 0; ch < numChunks; ch++) {
        uint mBase = ch * TS;

        // per-token decay/gate and prefix products:
        //   alpha_m = sigmoid(-a_m), beta_m = sigmoid(b_m)
        //   P_m = prod_{p<m} alpha_p, w_m = beta_m / P_{m+1}
        if (tid == 0) {
            for (uint m = 0; m < TS; m++) {
                float a = aRaw[(mBase + m) * 16 + qk];
                float b = bRaw[(mBase + m) * 16 + qk];
                alpha_sh[m] = 1.0 / (1.0 + exp(a));
                beta_sh[m] = 1.0 / (1.0 + exp(-b));
            }
            alpha_P[0] = 1.0;
            for (uint m = 0; m < TS; m++) alpha_P[m + 1] = alpha_P[m] * alpha_sh[m];
            for (uint m = 0; m < TS; m++) w_sh[m] = beta_sh[m] / alpha_P[m + 1];
        }
        barrier();

        // stage K tile (Q/V/G read from global later)
        for (uint i = tid; i < TS * DIM; i += TS * TS) {
            uint m = i / DIM;
            uint d = i % DIM;
            Ks[m][d] = K[(mBase + m) * (16 * DIM) + qk * DIM + d];
        }
        barrier();

        // 16x16 GEMM over DIM: Dk[j][m] = K_j.K_m, Dq[j][m] = K_j.Q_m
        {
            float dk = 0.0;
            float dq = 0.0;
            uint qoff = (mBase + c) * (16 * DIM) + qk * DIM;
            for (uint d = 0; d < DIM; d += 4) {
                vec4 kj = vec4(Ks[r][d], Ks[r][d + 1], Ks[r][d + 2], Ks[r][d + 3]);
                vec4 km = vec4(Ks[c][d], Ks[c][d + 1], Ks[c][d + 2], Ks[c][d + 3]);
                dk += dot(kj, km);
                vec4 qm = vec4(Q[qoff + d], Q[qoff + d + 1], Q[qoff + d + 2], Q[qoff + d + 3]);
                dq += dot(kj, qm);
            }
            Dk[r][c] = dk;
            Dq[r][c] = dq;
        }
        barrier();

        // sequential scan over the 16 tokens (unavoidable: delta_m depends on delta_{j<m})
        for (uint m = 0; m < TS; m++) {
            if (tid < DIM) {
                uint i = tid;
                float sk = 0.0;
                float sq = 0.0;
                uint qoff = (mBase + m) * (16 * DIM) + qk * DIM;
                for (uint d = 0; d < DIM; d += 4) {
                    vec4 sv = vec4(S[h * (DIM * DIM) + i * DIM + d],
                                   S[h * (DIM * DIM) + i * DIM + d + 1],
                                   S[h * (DIM * DIM) + i * DIM + d + 2],
                                   S[h * (DIM * DIM) + i * DIM + d + 3]);
                    vec4 km = vec4(Ks[m][d], Ks[m][d + 1], Ks[m][d + 2], Ks[m][d + 3]);
                    sk += dot(sv, km);
                    vec4 qm = vec4(Q[qoff + d], Q[qoff + d + 1], Q[qoff + d + 2], Q[qoff + d + 3]);
                    sq += dot(sv, qm);
                }
                skv[i] = sk;   // (S0 . K_m)[i]
                sqv[i] = sq;   // (S0 . Q_m)[i]
            }
            barrier();

            if (tid < DIM) {
                uint i = tid;
                float p = 0.0;
                float qc = 0.0;
                for (uint j = 0; j < m; j++) {
                    float wj = w_sh[j];
                    float dj = delta[j][i];
                    p  += wj * Dk[j][m] * dj;
                    qc += wj * Dq[j][m] * dj;
                }
                float vm = V[(mBase + m) * (32 * DIM) + h * DIM + i];
                float dv = vm - alpha_P[m] * skv[i] - alpha_P[m] * p;
                delta[m][i] = dv;

                float ym = alpha_sh[m] * (alpha_P[m] * sqv[i] + alpha_P[m] * qc) + beta_sh[m] * dv * Dq[m][m];
                float gm = G[(mBase + m) * (32 * DIM) + h * DIM + i];
                yGated[(mBase + m) * (32 * DIM) + h * DIM + i] = ym * (gm / (1.0 + exp(-gm)));
            }
            barrier();
        }

        // state update: S = P_16 * (S0 + sum_m w_m * delta_m * K_m^T)
        for (uint idx = tid; idx < DIM * DIM; idx += TS * TS) {
            uint i = idx / DIM;
            uint d = idx % DIM;
            float acc = 0.0;
            for (uint m = 0; m < TS; m++) {
                acc += w_sh[m] * delta[m][i] * Ks[m][d];
            }
            float s_old = S[h * (DIM * DIM) + i * DIM + d];
            S[h * (DIM * DIM) + i * DIM + d] = alpha_P[TS] * (s_old + acc);
        }
        barrier();
    }
}
```

**Chunked recurrence derivation** (matches the sequential `deltanet_ref` exactly, up to float summation order). Sequential:

```
delta_m = V_m − S_m·K_m
y_m     = alpha_m·(S_m·Q_m) + beta_m·delta_m·(K_m·Q_m)
S_{m+1} = alpha_m·S_m + beta_m·delta_m·K_m^T
```

Unrolling the state over a chunk starting at S₀:

```
S_m·K_m = P_m·(S₀·K_m) + P_m·Σ_{j<m} w_j·(K_j·K_m)·delta_j
S_m·Q_m = P_m·(S₀·Q_m) + P_m·Σ_{j<m} w_j·(K_j·Q_m)·delta_j
S_16    = P_16·(S₀ + Σ_m w_m·delta_m·K_m^T)
```

where `P_m = Π_{p<m} alpha_p` and `w_m = beta_m / P_{m+1}` — the `w` weighting keeps every term bounded even though `P_m` decays geometrically. The two 16×16 dot matrices (`Dk`, `Dq`) are classic 16×16-tiled GEMMs over dim 128; the triangular scan over the 16 tokens is sequential (state dependency) but parallel across the 128 dims.

---

### 7.4 FFN (SwiGLU)

#### `RmsNorm-swiglu-ffn-*` / `-GEMM-*`

RMSNorm → two parallel GEMMs (gate, up) sharing the same A tile → `y = silu(gate) · up`:

```glsl
gateResult /= (1.0 + exp2(-gateResult * 1.44269504));
y[globalCol] = gateResult * upResult;
```

GEMV version: 256 threads, one output column each, ping-pong shared buffers for the scale/zero tiles (INT8/INT4). GEMM version: 16×16 tiled (per-row RMS reduction over 16 token rows, dual B tiles `BgSub`/`BuSub`, direct per-tile scale/zero indexing — no ping-pong needed).

---

### 7.5 Prototypes (not used by validation)

`gemv1..7.comp` / `gemv.comp` (experimental GEMV variants), `gemm.comp` (naive scalar 16×16 tiled GEMM), `online-softmax.comp` (standalone online softmax + V accumulation, still validated via `validateOnlineSoftmax`), `RmsNorm.comp` (standalone RMSNorm), `RmsNorm-GEMV.comp` (fp32, unused), `RmsNorm-GEMV-Rope-*.comp` (early rope prototypes, superseded by RmsNorm-QKV), `test.comp`.

### 7.6 Inference shaders (engine mode)

The engine mode introduced a second generation of GEMM kernels ("GEMM2") plus dedicated decode split-K passes. All GEMM2 kernels consume `st->invRms` computed once per chunk by `RmsNorm-Prologue`, and use register-blocked workgroups (each thread accumulates 2–4 output columns via `TN`-wide B tiles) with direct per-tile scale/zero indexing.

#### `RmsNorm-Prologue.comp` (prefill, all GEMM2 kernels)

One 64-thread workgroup per token row. Each lane strided-sums `dot(v,v)` over the row, `subgroupAdd` reduces the 64 lanes, `subgroupElect` writes `invRms[m] = inversesqrt(sum/K + 1e-5)`. Bindings: `0 = x`, `1 = invRms` (a **write-only** scratch buffer, not gamma). Push `{K}`. Cost at M=512: ~0.4 ms — it replaces the per-layer RMSNorm reductions that every fused GEMM2 kernel previously re-ran.

#### `GEMM-ADD2-*` (prefill, down/out projections with residual)

`C = A·B + R`. INT4 uses `TN = 64` with 4 accumulators/thread; INT8/FP16 use `TN = 32` with 2. **All three variants are ping-pong double-buffered**: k-tiles for `t+1` are loaded into the alternate `Asub[2][4][TS]` / `Bsub[2][4][TN]` LDS set while tile `t` computes, halving the barriers (one per tile instead of two). B-load for INT4: `tid < 64` threads load 64 columns × one `uvec4` per k-tile (parity-selected nibble half), dequantized with `scale/zero` indexed `g*(K/4) + t*4 + kv`. The residual `R` is read once at the end, not accumulated per tile. Used by `addGemmAdd` for the FFN down projection (`k = FFN_N` when the input is `act`) and the attention `out` projection (`k = K`). When `m == 1` (decode) `addGemmAdd` routes to the split-K GEMV path instead. Ping-pong measured −2.4% (INT4), −1.4% (INT8), −0.2% (FP16).

#### `Gate-Sigmoid` (full-attention output gate)

Precision-agnostic elementwise pass: `attnOut *= sigmoid(gAttn)` per token over the 4096 attention output columns, 256 threads, push `{count}`. Inserted once per full-attention layer after the attention output (`Att-PV2` prefill, `Reduce-Att2` / `Att-full` decode) and before the `o_proj` residual GEMM.
#### `RmsNorm-QKV-GEMM2-*` + `Rope-GEMM-*` (prefill, split QKV projection)

The fused prefill QKV-GEMM (head-wide 256-wide B tiles, RoPE inside) was split into two passes sharing a raw-projection scratch buffer `st->qkvRaw` (`maxM × MODEL_QKV_N` floats):

- **`RmsNorm-QKV-GEMM2-{FP16,INT8,INT4}`** — prologue-style register-blocked GEMM (`TN = 64`/4 accs for INT4, `TN = 32`/2 accs otherwise) computing the *raw* q/k/v projections into `qkvRaw`; no norms, no RoPE, no cache writes. Bindings: `x, gammaIn, w, [scale, zero], qkvRaw, invRms`; push `{M, N, K}`.
- **`Rope-GEMM-{FP16,INT8,INT4}`** — grid `(2*heads + 2*kv_heads, M)`, 256 threads; each workgroup reads one token row of one head from `qkvRaw[row*N + head*256 + col]`. Routing by workgroup id: q heads (0..15) apply per-head RMSNorm scaled by the learned `q_norm[256]`, then RoPE with `angle = (tokBase + row) * theta[col & 127]` and write to `qOut`; g heads (16..31) store the raw gate projection to `gAttn` (no norm/rope); k heads (32..35) apply RMSNorm scaled by `k_norm[256]`, RoPE, then store to the cache (fp16 pack, or `uint8` + scale/zero at the fixed slot `kvh*MODEL_MAX_CTX + token`); v heads (36..39) store raw to the cache. The learned norm gammas are applied to each column **before** the RoPE rotation. Push `{N, gOffset, kOffset, vOffset, tokBase}`.

This cut the QKV chain from ~24.6 s to ~8.7 s per 16k prompt (fused avg ≈96 ms/call → GEMM2 ≈25–42 ms + Rope ≈0.3–0.8 ms). It also removed the prefill's dependence on the position buffer entirely (see §8.2).

#### `RmsNorm-LinearProj-GEMM2-*` (prefill, delta-net input projection)

Prologue + GEMM2 with `N = 12352`, `TN = 32`. Writes directly into the six routed output buffers (q/k/v/g/a/b) via the per-column stride layout `[m][qk*N_QK*DIM + d]`, `[m][qk]`, etc. (same routing as the old fused shader). A ping-pong probe on this family **regressed** (+1.5% INT4) — small-N grids don't have enough parallelism to hide the prefetch (gotcha 16).

#### `Embed-Gather.comp` (prefill, layer-0 embedding pre-fetch)

Replaces `Embed-RmsNorm-LinearProj-GEMM-FP16` in the engine prefill: the tied-lm-head column fetch is hoisted into its own precision-agnostic gather pass. Grid `(M, 1)`, 256 threads; each thread gathers vec4 chunks `i = col, col+256, …` of the embedding row `tokenIds[row]`: vec4 index `i` reads `lm[(i/2)*V + tok]`, even `i` unpacks `.x,.y`, odd `i` unpacks `.z,.w`. Writes the fp32 row to **both** `st->embStaged` and `st->embOut` (the residual copy). Bindings: `tokenIds, lm, embStaged, embOut`; push `{V}`; ~0.42 ms/call. The projection then runs as plain `RmsNorm-Prologue` + `RmsNorm-LinearProj-GEMM2-FP16` over the contiguous `embStaged` (via `addLinearProj(..., st->embStaged)`), avoiding the column-major strided fetch inside the GEMM: embed path 8.5 s → ~2.1 s per 16k prompt.

#### `RmsNorm-swiglu-ffn-GEMM2-FP16` (prefill, FP16 swiglu — kept dual-matrix)

Prologue + dual B tiles (`gate` and `up`, `TN = 32`), silu applied inside the kernel: `y = silu(gateResult) * upResult`. This is the **FP16** variant; INT4/INT8 were replaced by the flat kernel below. Also ping-pong double-buffered (`Asub[2][4][TS]`, `BgSub[2][4][TN]`, `BuSub[2][4][TN]`) — 91.5 → 89.7 ms/call.

#### `RmsNorm-swiglu-flat-GEMM2-INT4/INT8` + `Swiglu-combine.comp` (prefill, flattened swiglu)

The INT4/INT8 swiglu kernels. The grid is flattened over `2×FFN_N` columns: workgroup x-tile `nBase < off` computes gate columns, `nBase >= off` computes up columns (`off = FFN_N`, routed by `upHalf`). Each workgroup loads from **one** weight matrix (gate or up) with a single-matrix B-loader — no dual-matrix staging, no silu inside the inner loop. Output goes to `st->gAct` / `st->uAct` (fp32, `maxM×FFN_N`). Both flat kernels are ping-pong double-buffered over k-tiles (~−1..2%; INT4 110.6 ms, INT8 107.4 ms at M=512). `Swiglu-combine.comp` (256 threads, `dispatchX = m*N/256`) then computes `act = silu(gAct) * uAct` elementwise (~0.5–0.7 ms per call; total < 1 s per 16k prompt). FP16 was measured *slower* flat (103.8 vs 91.4 ms) and keeps the dual kernel; TN=64 and TS=32 probes on the flat kernels regressed +50–70% and were reverted (gotcha 16).

#### `Att-QK2-*` / `Att-Softmax.comp` / `Att-PV2-*` (prefill, unfused attention)

Attention was split into three passes over a persistent fp16 score buffer `st->attScores` (`maxM × MAX_CTX × 4` bytes per head-quad row, 134 MB at M=512):

- **QK2** (`Att-QK2-*`): workgroup = (head-quad, m-tile); 16 q-tokens × 64 kv-tokens; TN=64 B tiles over the head dim, K staged per head-quad lane; causal limit `limit = qOff + mGlobal` is **chunk-absolute** (correct across chunked prefill; the legacy `Att-full-GEMM-*` masked with the chunk-local row and under-attended history). Writes `NEG_INF` outside the causal range into the fp16 score buffer.
- **Softmax**: 256 threads per (m-tile, head-quad) row; strided max → `subgroupMax` → cross-subgroup max via 4 shared slots, then `exp(s - gmax)` in place. The final normalize sweep was **folded into PV2**: Softmax writes `smSum[hL*param.mRows + row] = 1/total` (binding 1) and PV2's epilogue scales each accumulator (`oacc *= smSum[hL*mRows + mGlobal]`; binding 3 for FP16, binding 5 for INT8/INT4). This removed a third full pass over the score buffer: softmax pool 3.2 s → ~0.5 s.
- **PV2**: workgroup = (head-quad, m-tile, v-tile); p-tile staged as `p_sh[16][17]` (padded LDS to avoid bank conflicts), V staged per kv-token; 16 threads each accumulate 16 output dims per token (`oacc += p_sh[row][kk] * Vsub[col][kk]`), 64 threads per tile — padded `Vsub[TS][TS+1]` staging. Quantized INT4/INT8 V dequantized during staging; epilogue multiplies by the softmax reciprocal from `smSum`.

Whole-attention cost dropped 48.7 s → ~10.4 s for a 16k prompt (QK2 ≈ 8.2 s, SM ≈ 0.5 s, PV2 ≈ 1.7 s).

#### Decode split-K family (M=1)

- **`GEMV-SplitK-*` + `Reduce-GEMV-ADD.comp`**: the K dimension is split across `dispatchY = 4` workgroups (256 threads each, strided k-tiles), partial sums written to `st->gemvPartial[chunk*N + col]`; `Reduce-GEMV-ADD` sums the 4 partials and adds the residual (`C[g] = Σ_z P[z*N+g] + R[g]`). Used by `addGemvSplit` for attention-out and FFN-down when `m == 1`.
- **`RmsNorm-up-ffn-SplitK-*` + `FFN-Down-SplitK-*`**: decode FFN flattened like the prefill flat kernel — one shader computes `silu(gate)·up` over `2×FFN_N` columns (routing via `nBase >= off`, push `{M,N,K,off}`), writing `st->ffnPartial`; `FFN-Down-SplitK` then GEMVs the down matrix over the partial (which `Reduce-GEMV-ADD` reduces into `h`).
- **`RmsNorm-QKV-SplitK-*` + `Reduce-Rope-*`**: split-K QKV projection to `st->qkvPartial`, then a per-head reduce that routes by column ranges — g (raw to `gAttn`), q/k (RMSNorm scaled by learned `q_norm`/`k_norm` + RoPE), v (raw) — and stores q/k/v to cache (replaces the old fused `RmsNorm-QKV-*`). Quantized scale/zero are written at the fixed slot `kvHead * MODEL_MAX_CTX + pos`.
- **`RmsNorm-LinearProj-SplitK-*` + `Reduce-LinearProj.comp`**: split-K proj to `st->linprojPartial`; reduce routes the 12352 columns into the six q/k/v/g/a/b output buffers.
- **`Att-SplitK2-*` + `Reduce-Att2.comp`** (decode attention, ctx ≥ 256): `Att-SplitK2` splits the KV sequence into up to `MAXC = 128` chunks of 4 KV-tiles (LOOPS=4, 64-token tiles); each workgroup runs an online-softmax over its chunk and writes `{max, sum, acc[HEAD_DIM]}` per (chunk, head) into `st->attPartial`; `Reduce-Att2` re-normalizes across chunks (`w = exp(P[ml*2] - m)`). Below 256 tokens the engine uses `Att-full-*` (single workgroup, online softmax, §7.2).

---

## 8. Inference Engine (src/generate.c)

The engine compiles the model into `operation` arrays at startup and executes them with the split record/submit/wait API from §3.3 (`executeRecord` → `executeSubmitNow` → `executeWaitLast` → `logLastFrame`), which keeps per-op GPU timestamps and avoids the per-op fence stall of `execute()`.

### 8.1 Op compilation

`addOp` appends a fully-formed `operation` (shader name, buffers, push constants, dispatch). The layer builders (`buildFfn`, `buildAttention`, `buildDelta`, `buildLinearProj`, `buildLmHead`) emit the op sequences from §7.6. `createGenerator` pre-compiles four arrays:

- **`groupOps` / `groupOpsShort`** — decode group: `DECODE_GROUP = 4` tokens × (Embed-LinearProj → 32 layers → lm-head). The two variants differ only in attention: split-K (`Att-SplitK2`+`Reduce-Att2`) vs `Att-full`, selected by `ATT_SPLIT_THRESHOLD = 256` on the current context length.
- **`prefillOps`** — one chunk of `maxM` tokens (`MODEL_PREFILL_CHUNK`): Embed-Gather → RmsNorm-Prologue → LinearProj-GEMM2-FP16 (over `st->embStaged`) → 32 layers → (no lm head).
- **`finalOps`** — lm head over `st->lastRow` (the last prefill row copied back to host and re-uploaded), `doIncrement = 0`.

### 8.2 Chunked prefill (`runPrefill`, `compilePrefill`, `executeChunked`)

The prompt is processed in chunks of `MODEL_PREFILL_CHUNK` (512) tokens:

1. Copy the chunk's token ids into `st->tokenIds` (mapped RAM).
2. `compilePrefill(g, cur, offset)` re-emits the whole prefill op list for the chunk's actual row count `cur` (layers get `m = cur`, so dispatchY shrinks for the tail chunk). Chunk offsets are threaded **at build time** — `buildLayer` / `buildAttention` take the chunk's absolute token base and emit it directly into push constants (`Rope-GEMM tokBase`, attention `{ctxLen = offset + m, qOff = offset}`); there is no post-compile patch loop. Prefill shaders never touch the position buffer; after the last chunk's ops are recorded, `runPrefill` sets it host-side via `stateSetPosition(g->s, &g->st, nextPos + nTokens)` before `finalOps` executes.
3. `executeChunked` submits the op list in slices of ≤ 8 ops, waiting between slices — this keeps GPU work below the Windows TDR threshold (`TdrDelay = 8 s`; an over-long slice yields `VkResult -4` = `VK_ERROR_DEVICE_LOST`).
4. After the last chunk, the last processed row is copied from `st->h` to `st->lastRow` (host round-trip, only once per prefill), and `finalOps` selects the first generated token.

### 8.3 Decode loop (`generateTokens`) and the server

- `generateTokens(g, prompt, nPrompt, maxNewTokens, out, &outCount)` = `runPrefill` + the decode loop, writing generated ids to a caller buffer and **stopping at EOS** (`token == g->eos`). `maxNewTokens` is clamped so `nPrompt + maxNewTokens <= g->maxCtx`.
- The op lists are **double-buffered**: while group `u` executes, group `u+1` is recorded (the current token is written into `tokenIds[0]` after each group's result readback). Two separate op arrays (`groupOps` vs a shadow copy compiled with `passIdx`-tagged lm-head write slots) are alternated each iteration.
- `resetGenerator(g)` re-zeroes the 24 gated-delta `stateS` recurrence buffers (`createTransferAndCopy` re-copies their still-zero staging buffers) and resets `nextPos = 0`, so each server request is an independent full prompt.
- `serverMain` (compute.c) sets binary stdio (`_setmode`), parses `--weights/--vocab-head/--vocab-embed/--eos/--max-ctx/--max-new`, creates the session+generator once, then loops: read `n` prompt ids → `resetGenerator` → `generateTokens` → write `m` output ids → until `n == 0`.
- The group's `ArgMax-Reduce` increments the shared position buffer once per token (`doIncrement = 1`); prefill never touches it on the GPU (`runPrefill` sets it host-side after the last chunk), and the lm-head final pass runs with `doIncrement = 0` — together these keep the position counter correct across the prefill→decode transition (see gotcha 8).

### 8.4 Timing log

The dispatch layer retains per-op timestamp logging (`logFrame`/`logLastFrame` and the `timing_agg`/`timing_log.txt` machinery in `dispatch.c`), gated by `setTimingEnabled`. It is currently **not wired to any CLI flag** (the old `main.exe log` mode was dropped when the engine became a persistent server); `timingEnabled` stays 0 so the decode loop runs with no per-op logging.

---

## 9. Key Gotchas Discovered

1. **k-tile loop bound is `K / TS`, not `K/(TS·vec)`.** Each iteration advances k by 16 floats (4 vec4s), so the loop runs K/16 times.
2. **GatedDeltaNet `alpha`/`beta` are per-token** (`aRaw[m*16+qk]`, not `aRaw[qk]`). The recurrence is non-stationary; the chunked form needs prefix products `P_m` and weights `w_m = beta_m/P_{m+1}`.
3. **Flash-attention online rescale:** per-element probability must be `exp(score − tile_max)` (not `− m_next`) or `beta` gets applied twice; the sum reduction clobbers the shared array, so per-element probabilities live in a separate `s_p` buffer.
4. **GatedDeltaNet is numerically unstable** with the harness's random weights (`−beta·k·kᵀ` has a large eigenvalue, so the state overflows within ~16 tokens). The `validateGatedDeltaNetGEMM*` functions scale the projection weights by 1/64 (exact in fp16 — exponent shift) so the recurrence stays stable over M=64 and the chunked-vs-sequential comparison is meaningful.
5. **Windows make quirks:** recipes run through `cmd.exe` — `mkdir`/`if exist` paths must use backslashes (`/` is treated as a switch prefix), and shader folder names must not contain spaces (make word-splits `$(wildcard)` output).
6. **KV cache layout is `[token][row]`** (transposed relative to the host `[row][seq]`), so attention shaders index `cache[token*rows + row]`.
7. **Workgroup-wide barriers need every thread present.** For GEMV shaders with a partial last workgroup (N not divisible by 256), out-of-range threads must not early-return — they still run the loop/barriers and carry a `-inf` (`uintBitsToFloat(0xFF800000u)`) accumulator so they can never win the argmax reduction.
8. **Position buffer write rules (multi-layer safety).** The shared `uint[1]` position buffer must be updated exactly once per phase transition: decode's per-token `+1` lives **only** in `ArgMax-Reduce` (single dispatch, gated by `doIncrement`), and prefill sets it host-side (`stateSetPosition`) after the last chunk — no prefill shader writes it at all now. Historically the fused QKV GEMM wrote it absolutely (`tokenIdx + M`, idempotent across layers); a read-modify-write `+= M` in any per-layer shader would inflate the counter N× per prefill chunk, and an unconditional `+1` in the lm-head step would corrupt the prefill→first-decode transition.
9. **`float atomicAdd` on SSBO is a no-op** on this driver stack. Split-K float accumulation must use `atomicCompSwap` CAS (or the two-pass partial+reduce pattern used here).
10. **The `RmsNorm-Prologue` output binding is `invRms`, not gamma.** A bug bound `{h, gammaF}`/`{h, gammaIn}` and wrote invRms into the weight gamma while `st->invRms` stayed 0 — every GEMM2 projection computed zeros end-to-end (silently, because the demo collapses to token 0 and the validators use their own wiring). Timings were unaffected (same FLOPs). Fix: `proBufs[] = {st->h, st->invRms}`.
11. **Causal masks must use chunk-absolute indices in chunked prefill.** `Att-full-GEMM-*` masked with the chunk-local row (`kvTok <= row`), under-attending history; the QK2 kernels use `limit = qOff + mGlobal`.
12. **Build from the repo root; headers aren't dep-tracked.** Running `make` from `bin/` silently no-ops (no Makefile there). The Makefile compiles `.c`→`.o` with no `.d` dependency generation, so **any** header edit leaves stale `.o` files compiled against the old layout. This bit twice: after `model.h` dim edits, and much worse when a field was added to the `buffer` struct (`buffer.h`) — the not-recompiled `generate.c`/`dispatch.c` kept the old sizeof, corrupted the ABI, and produced a deterministic `0xC0000005` crash mid-weight-load that looked like the transient gotcha 14. **Rule: after editing any header, `make clean && make`.** If shader files were restored with `Copy-Item`, delete the stale `bin/shader/<name>.spv` — timestamps are preserved and the rebuild is skipped.
13. **Chunked execution avoids TDR.** Submitting one giant prefill command buffer (16k tokens) exceeds the Windows 8 s TDR budget (`VK_ERROR_DEVICE_LOST = -4`). Slice submissions to ≤ 8 ops with a wait between slices.
14. **Long-running jobs can transiently crash in `0xC0000005` during weight load** (weight file ~2 GB); retrying the run succeeds — not a code bug, worth re-running before debugging. Validation runs can also transiently produce NaN/garbage in one shader; two consecutive green vals = real pass.
15. **Quantized KV-cache scale/zero need a context-independent stride.** Writers originally used `kvh*(pos+1)+pos` / `kvh*(tokenIdx+M)+absTok` — a stride that grows with the current context — while readers assumed whatever the *current* chunk's stride was, so every token written by an earlier chunk had its scale read from the wrong slot once the cache grew (silently, and masked by the zero-token demo signature). Fix: fixed stride `kvh * MODEL_MAX_CTX + token` in all writers (`Rope-GEMM`, `Reduce-Rope`) and readers (`Att-QK2/PV2/SplitK2/full` INT8+INT4), with validator fixtures updated to match.
16. **GCN4 tile geometry for the GEMM2 family: smaller workgroups win.** TN=64 or TS=32 (1024-thread) variants regress +50–70% despite halving barrier count — occupancy/latency-hiding dominates. Ping-pong k-tile prefetch (one barrier per tile, prefetch into the alternate LDS set) gives a reliable but small −1..2% **only when grid.x is wide** (FFN/ADD2 shapes); on narrow-N kernels (LinearProj N=12352, QKV) it regressed or washed out.
17. **Appended shader bindings must match validator buffer order exactly.** When the gated-attention change added `q_norm`/`k_norm`/`gAttn` bindings, the three *legacy prefill-fused* `RmsNorm-QKV-GEMM-*` shaders declared them as `gAttn, qGamma, kGamma` while the validators supplied `qGamma, kGamma, gOut` — q got the wrong gamma and g/k wrote nowhere (q err 6.4, g/k zero). The decode-fused and `Rope-GEMM`/`Reduce-Rope` shaders matched, which is why only `validateQkvRopeGEMM*` failed. Also: the learned per-head norm gamma must be applied to each column **before** the RoPE rotation mixes the two halves (`acc*inv_rms*gamma` then rope), not to the rotated result — applying it after yields a real numeric mismatch against the reference, not just noise.
18. **VRAM is ~7936 MB on the 8 GB RX 580, not 8192, and it's fragmented.** `main.exe meminfo` dumps this: `heap[0]` device-local is 7936 MB (WDDM reserves ~256 MB) and host-visible/staging memory lives in a separate 16/32 GB system heap — staging is *not* the VRAM problem. The 9 B model + state needs ~7480 MB of device-local memory, and because ~450 discrete `vkAllocateMemory` calls fragment the heap, the allocation that tips it over is reproducible: `OOM: vkAllocateMemory failed for 'attScores' (DEVICE_LOCAL, 128.00 MB) | device_local=7352.66 MB`. `createBufferNamed(..., name)` + per-pool byte counters (reset) in `buffer.c` emit this line and name the buffer. Freeing device memory = lower `MODEL_MAX_CTX` / smaller `MODEL_PREFILL_CHUNK`, INT8 embed/lm-head, or consolidating the ~450 tiny allocations into arenas.

---

## 10. Verification Results (M=64, max absolute error vs CPU reference)

| Shader | max_err | GPU time |
|---|---|---|
| GEMM-FP16 / INT8 / INT4 | 0.000080 / 0.000076 / 0.000080 | ~41 ms |
| GEMM-ADD FP16 / INT8 / INT4 | 0.000069 / 0.000069 / 0.000071 | ~14 ms |
| RmsNorm-swiglu-ffn-GEMM FP16 / INT8 / INT4 | 0.0129 / 0.0176 / 0.0142 | 60–77 ms |
| RmsNorm-LinearProj-GEMM FP16 / INT8 / INT4 | 0.00018 / 0.00020 / 0.00021 | 43–51 ms |
| QKV-Rope-GEMM FP16 (q / k-cache / v-cache) | 0.000014 / 0.0021 / 0.062 | ~56 ms |
| QKV-Rope-GEMM INT8 (q / g / k-cache / v-cache) | 0.000010 / 0.0003 / 0.011 / 0.306 | ~130 ms |
| QKV-Rope-GEMM INT4 | 0.000014 / 0.014 / 0.32 | ~51 ms |
| QKV-Rope-GEMM-pos FP16 / INT8 / INT4 (position buffer = M) | 64 / 64 / 64 exact | included above |
| QKV-Rope-pos FP16 / INT8 / INT4 (decode read, unchanged) | 33 / 33 / 33 exact | included above |
| Attention-GEMM FP16 / INT8 / INT4 | 0.000003 / 0.000003 / 0.0017 | ~0.7 ms |
| GatedDeltaNet-GEMM FP16 (out / state S) | 0.000065 / 0.000005 | ~57 ms |
| GatedDeltaNet-GEMM INT8 | 0.000061 / 0.000005 | ~64 ms |
| GatedDeltaNet-GEMM INT4 | 0.000103 / 0.000006 | ~67 ms |
| Gate-Sigmoid | 0.000000 | ~0.02 ms |
| LMHead-ArgMax FP16 (token index) | exact match | ~117 ms (4096 × 81920 GEMV + reductions) |
| LMHead-ArgMax-pos FP16 (position buffer, +1) | 41 → 42 exact | included above |
| Embed-RmsNorm-LinearProj FP16 (q / k / v / g / a / b) | 0.000164 / 0.000145 / 0.000168 / 0.000175 / 0.000088 / 0.000084 | ~2.6 ms |
| Embed-RmsNorm-LinearProj-GEMM FP16 (q / k / v / g / a / b) | 0.000381 / 0.000313 / 0.000290 / 0.000381 / 0.000198 / 0.000252 | ~155 ms |

The larger k/v-cache errors are fp16/int4 cache quantization, matching the existing GEMV baselines. Note the CPU references are single-threaded and dominate total runtime (`main.exe val` takes several minutes); the GPU shader times above are per-shader query-pool timestamps.

The table above is the **validation harness** (M=64). The engine additionally self-checks end-to-end with its synthetic (seeded-random) weights: a 16k-token run must produce the expected deterministic signature — currently an all-zero 128-token stream (`count=128, unique=0`, token 0 repeated) — and `main.exe val` must stay green (only the pre-existing large-magnitude `GatedDeltaNet` signatures may appear). A clean 16k logged run is the acceptance test for any prefill shader change.

---

## 11. Build & Run

```bash
make clean          # removes bin/ and build/  (required after ANY header edit — see gotcha 12)
make                # compiles all shader/**/*.comp -> bin/shader/*.spv, builds bin/main.exe

cd bin && main.exe val       # validation harness (every validate* + max_err + timing)
cd bin && main.exe meminfo   # dump memory heaps/types (device-local vs host-visible) and exit
```

Real weights are loaded by the **server** (`main.exe`, default). It reads a length-prefixed prompt from stdin and writes generated ids to stdout, so it is normally driven by the Python frontend:

```bash
uv venv .venv                                  # once
uv pip install --python .venv/Scripts/python.exe tokenizers   # once

.venv/Scripts/python.exe vk_llm.py <weight_dir> <max_ctx> [<custom_vocab_dir>] ["prompt text"]
```

`vk_llm.py` exposes `start_llm(weight_dir, max_ctx, vocab_weight=None, max_new_tokens)`, `tokenize`, `generate`, `detokenize`, `close`. `vocab_weight=None` uses the full 248320 vocab (tokenizer + shard lm-head/embed); a custom vocab dir (e.g. `model/Qwen3.5-pruned-vocab`) selects the pruned tokenizer and the `lm_head.*.safetensors`/`embed_tokens.*.safetensors` discovered in `weight_dir`. EOS = the tokenizer's `<|im_end|>` id (81896 pruned / 248046 full).

---

## 12. VRAM budget (8 GB RX 580)

Device-local memory (`heap[0]`) is **7936 MB**, not 8192. With the current hybrid spec the totals are (metered by `createBufferNamed` + the `OOM:` line in `buffer.c`):

| Component | MB |
|---|---|
| weights (6545.74) | 6545.74 |
| KV k/v cache (8 full-attn layers) | 640 |
| KV scale/zero (6 INT8/INT4 layers) | 12 |
| `attScores` (`maxM × MAX_CTX × 8`) | 128 |
| `act` + `gAct` + `uAct` | 72 |
| `stateS` delta recurrence (24 × 2) | 48 |
| h/emb/attn/q-proj group, `qkvRaw`, partials | ~155 |
| **total device-local** | **~7550** |

This fits arithmetically but overflows at runtime — fragmentation from ~450 discrete allocations means a single further 128 MB `attScores` block can't be satisfied. The failure is deterministic and reported as `OOM: vkAllocateMemory failed for 'attScores' (DEVICE_LOCAL, 128.00 MB) | device_local=7352.66 MB`. Levers: lower `MODEL_MAX_CTX` (KV + attScores scale with it → −576 MB at 8192), lower `MODEL_PREFILL_CHUNK` (−160 MB at 256), INT8 embed/lm-head (−640 MB), or consolidating the ~450 tiny allocations into arenas. Staging/host buffers are **not** the issue — they live in the 16/32 GB system heap.

---

## 13. Known issues / not yet implemented

1. **Gated-deltaNet `conv1d` (and `A_log`/`dt_bias`/q-k `norm`) are not implemented.** `GatedDeltaNet.comp`/`-GEMM` run the simplified sigmoid-`a`/`b` delta rule + `silu(g)` gate only. The real block's short causal `conv1d` (kernel 4 over qkv), per-head `A_log` decay, `dt_bias`, and 128-dim q/k `norm.weight` are loaded into `w.conv`/`w.aLog`/`w.dtBias`/`w.attnNorm` but unused. Until these are folded in, linear-attention layers are **not HF-identical** (they produce a coherent, deterministic stream, not the reference output).
2. **`rms_norm_eps` = `1e-5`** everywhere (shaders + `rms_norm_apply`), but the model config is `1e-6`.
3. **Partial RoPE (`0.25`)**: the engine rotates the full head; the model rotates only 64 of 256 dims.
4. **Final logit scaling** (if any) is not applied — verify against HF.
5. **VRAM OOM** (§12) blocks a full end-to-end generation on the 8 GB card at `max_ctx = 32768`.
6. **MTP and vision tensors are ignored** (text-only baseline).
