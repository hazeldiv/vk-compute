# VK Compute — Complete Technical Summary

A Vulkan-based GPU compute engine for running LLM inference (target: **Qwen3.5 9B** with **hybrid quantization** — INT4, INT8, and FP16 layers). The C harness generates randomized test data, transposes and uploads weights, dispatches GLSL compute shaders, and validates GPU output against single-threaded CPU reference implementations.

---

## 1. Project Overview

- **Language/stack:** C (harness), GLSL 450 (compute shaders), Vulkan 1.1, glslangValidator, GNU Make (MinGW-w64 / MSYS2 UCRT64), Windows.
- **Two inference phases:**
  - **Decode (token generation):** `GEMV` shaders — one token (M=1) × weight matrix.
  - **Prefill (first token / prompt processing):** `GEMM` shaders — the whole prompt sequence (M=64 tokens) × weight matrix in one dispatch.
- **Precision:** every layer family exists in FP16, INT8 (per-group 256 asymmetric quantization), and INT4 versions. Some shaders (GatedDeltaNet) are precision-agnostic — they operate on already-dequantized float projections.
- **Validation:** every shader has a CPU reference (`*_ref` functions) and a `validate*` function that runs the shader and compares via max absolute error.

### Model dimensions used by the harness

| Symbol | Value | Meaning |
|---|---|---|
| `d_model` / `K` | 4096 | hidden size / reduction dimension |
| `N` (FFN) | 12288 | FFN gate+up columns |
| `wo_n` | 4096 | delta-net out projection / residual size |
| `qkv_n` | 6144 | QKV = 16 q-heads + 2×4 kv-heads, each dim 256 |
| `heads`, `kv_heads`, `att_dim` | 16 / 4 / 256 | full attention geometry |
| `n_qk`, `n_v`, `dim` | 16 / 32 / 128 | gated delta-net geometry |
| `proj_n` | 12320 | delta-net input projection (Q,K,V,G,A,B) |
| `M` (prefill) | 64 | prompt tokens processed by GEMM shaders |
| `vocab_size` | 81920 | pruned lm-head vocab (= 320 × 256 full workgroups) |

---

## 2. Repository Structure

```
vk-compute/
├── Makefile                    # recursive shader build -> bin/shader/*.spv
├── include/                    # C headers
│   ├── buffer.h  data.h  descriptor.h  device.h  dispatch.h
│   ├── fence.h  pipeline.h  session.h  validation.h
├── src/
│   ├── main.c                  # calls compute()
│   ├── compute.c               # generates test data, runs all validate* calls
│   ├── session.c device.c buffer.c command.c fence.c   # Vulkan setup
│   ├── descriptor.c pipeline.c dispatch.c              # descriptors, pipelines, dispatch
│   ├── data.c                  # pseudo-random data, fp16 conversion, transpose_block16
│   └── validation.c            # CPU reference impls + all validate* functions
└── shader/
    ├── Utility/                # shared matmul primitives
    │   ├── FP16/ INT8/ INT4/   # GEMV-*, GEMV-ADD-*, GEMM-*, GEMM-ADD-*, RmsNorm-GEMV-*
    │   ├── FP16/               # LMHead-GEMV-ArgMax-FP16.comp
    │   └── ArgMax-Reduce.comp  # final argmax over per-workgroup maxes (precision-agnostic, at root)
    ├── Full-Attention/         # QKV projection + RoPE + full attention
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-QKV-*, Att-full-* (GEMV + GEMM variants)
    ├── Linear-Attention/       # gated delta-net
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-LinearProj-* + Embed-RmsNorm-LinearProj-* (FP16 only)
    │   ├── GatedDeltaNet.comp  GatedDeltaNet-GEMM.comp   (precision-agnostic, at root)
    ├── FFN/                    # swiglu feed-forward
    │   ├── FP16/ INT8/ INT4/   # RmsNorm-swiglu-ffn-*
    │   └── RmsNorm-swiglu-ffn.comp                        (fp32, at root)
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
```

- `MEMORY_RAM` = host-visible staging; `MEMORY_VRAM` = device-local.
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
- KV cache: `[token][row]` — fp16 (`uint16_t` + `packHalf2x16`) for FP16/INT8, or `uint8` + per-(kv-head, token) scale/zero for INT4.
- RoPE theta: `theta[i] = 1e6^(-i/(dim/2))`, length `dim/2 = 128`.

### 4.5 Push constants

- Generic GEMM/GEMV/FFN/LinearProj: `{M, N, K}`.
- QKV: `{M, N, K, tokenIdx, kOffset, vOffset, context_length}` (7 ints, ≤ MAX_PUSH_CONSTANTS = 8).
- Attention: `{context_length}`. GatedDeltaNet: `{M}`.
- LM head argmax: `{M, N, K}` (M=1, N=vocab) for the GEMV-ArgMax pass; `{vocabSize}` for ArgMax-Reduce.
- Embed-RmsNorm-LinearProj GEMV/GEMM: `{M, N, K, V}` (V = vocab size, selects the lm-head column used as the embedding).

### 4.6 Dispatch geometry

- GEMV shaders: `dispatchX = N/256` (one workgroup of 256 threads per 256 output columns), M=1.
- GEMM shaders: `dispatchX = N/16`, `dispatchY = M/16` (each workgroup computes a 16×16 output tile).
- QKV GEMM: `dispatchX = 24` (heads), `dispatchY = M/16` (head-wide 16×256 tile).
- Attention: `dispatchX = heads`, `dispatchY = seq/16`.
- GatedDeltaNet GEMM: `dispatchX = n_v = 32`, one workgroup per v-head.
- LMHead-GEMV-ArgMax: `dispatchX = (vocab+255)/256`; ArgMax-Reduce: single workgroup.

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

Run: `make run` (= `cd bin && main.exe`). `make clean` removes `bin/` and `build/` recursively.

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

`compute()` (src/compute.c) wires everything: it keeps the original M=1 GEMV validations and adds M=64 (`Mg = 64`) GEMM validations with an M-token input `inputM = getData(4321, Mg, K)`, plus the FP16 lm-head argmax validation over `vocab_size = 81920` (weights `lmHeadFP16 = getDataFP16(15001, K, vocab_size)`, ~640 MB).

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

One workgroup of 256 threads reduces the per-workgroup winners from the first pass to a single token id. Bindings: `0 = maxValue (float[])`, `1 = maxIndex (uint[])`, `2 = result (uint[])`; push constant `{vocabSize}`; `dispatchX = 1`. Full source:

```glsl
#version 450
#define TS 256

layout(local_size_x = TS, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Params { uint vocabSize; } params;

layout(set = 0, binding = 0) readonly restrict buffer MaxValueBuffer { float maxValue[]; };
layout(set = 0, binding = 1) readonly restrict buffer MaxIndexBuffer { uint maxIndex[]; };
layout(set = 0, binding = 2) writeonly restrict buffer ResultBuffer { uint result[]; };

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
    }
}
```

**How it works:** `numGroups = ceil(vocabSize/256)` (= 320 for vocab 81920); each thread strided-loads `maxValue[i]`/`maxIndex[i]` in chunks of 256 (`i += TS` loop), keeping the running best with the same smallest-index tie-break, then one shared tree reduction; thread 0 writes the final token id to `result[0]`. Both shaders are chained as two `operation`s in one `execute()` call (the dispatch harness inserts a write→read barrier between ops).

---

### 7.2 Full Attention

#### `RmsNorm-QKV-*` (decode, one token)

Workgroup = one head (256 columns). RMSNorm over K → project `N = 6144` columns (q 0..4095, k 4096..5119, v 5120..6143) → for q/k heads: per-head RMSNorm over 256 cols, then **RoPE**: `angle = tokenIdx * theta[col%128]`, first half `a·cos − b·sin`, second half `a·sin + b·cos` (a/b from opposite halves) → q written to `qOut`, k/v stored to the KV cache at slot `(context_length-1)`. INT4 additionally quantizes k/v per (head, token) with min/max over the 256 cols and writes scale/zero.

#### `RmsNorm-QKV-GEMM-*` (prefill, head-wide tile)

Workgroup = (head, 16-token block); output tile = 16 tokens × 256 columns; the 16×16 k-tiling from §7.1 inside. Full FP16 source:

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

Workgroup = one query head. Query staged in shared (256 floats). Online softmax over KV-token tiles of 256: per tile compute scores `q·k` (fp16 or dequantized int8/int4 keys), tile max via tree reduction, exp, tile sum, then rescale running accumulator with `alpha = exp(m_prev − m_next)` / `beta = exp(tile_max − m_next)` and accumulate `P·V` per output dim. Output `o = acc / l`.

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

RMSNorm + single GEMM over `N = 12320`, output routed by column with per-token strides: `q[2048]` (cols < 2048), `k[2048]` (< 4096), `v[4096]` (< 8192), `g[4096]` (< 12288), `a[16]` (< 12304), `b[16]` (else).

#### `Embed-RmsNorm-LinearProj-*` (token-embedding fused variant, FP16 only)

Replaces the `x` input with a token-id lookup: the token id(s) (a `uint` buffer, GEMV: 1 element, GEMM: `M` elements) index into the **tied lm-head buffer** — the same block-transposed FP16 `uvec4[]` layout as `LMHead-GEMV-ArgMax-FP16.comp` binding 1 — to fetch the 4096-dim embedding row, then RMSNorm + LinearProj proceed exactly as `RmsNorm-LinearProj-*`. This shader is meant to run right after the LM-head argmax (token selection) so the next decode step can start from the selected token's embedding without CPU readback.

- **Bindings (GEMV and GEMM):** `0 = tokenIds (uint[])`, `1 = lm (uvec4[], transposed lm head [K][vocab])`, `2 = gamma (vec4[])`, `3 = w (uvec4[], transposed w_in [K][12320])`, `4..9 = q/k/v/g/a/b out (float[])`. Push constants `{M, N, K, V}`.
- **Embedding fetch:** vec4 index `i` (0..1023) reads `lm[(i/2)*V + tok]`; even `i` takes `unpackHalf2x16(raw.x), unpackHalf2x16(raw.y)`, odd `i` takes `.z,.w` (each `uvec4` = 2 vec4 k-rows of a column).
- **GEMV (`Embed-RmsNorm-LinearProj-FP16.comp`):** 256 threads/col, subgroup RMSNorm sum identical to `RmsNorm-LinearProj-FP16`; `dispatchX = (12320+255)/256 = 49`.
- **GEMM (`Embed-RmsNorm-LinearProj-GEMM-FP16.comp`):** 16×16 tiled clone of `RmsNorm-LinearProj-GEMM-FP16`; per-token embedding fetched by `tokenIds[mGlobal]` in both the sum-sq pass and the A-tile staging; `dispatchX = 12320/16 = 770`, `dispatchY = M/16`.

Note the lm-head column fetch is strided by `V` uvec4s (column-major access into a 640 MB buffer), so the GEMM variant is slower than the plain `x`-input GEMM (~150 ms vs ~45 ms at M=64) — a layout artifact of tied embeddings, not a correctness issue.

#### `GatedDeltaNet.comp` (decode, one token)

Workgroup = one v-head; state `S` is a 128×128 matrix per head in VRAM. Per token:

- `alpha = 1/(1+exp(a))`, `beta = 1/(1+exp(-b))` — **per-token** (from the token's own `a`/`b` projections)
- `delta = V − S·K`
- `y = alpha·(S·Q) + beta·delta·(K·Q)`
- `yGated = y · silu(G)`
- state update: `S ← alpha·S + beta·delta·Kᵀ`

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

---

## 8. Key Gotchas Discovered

1. **k-tile loop bound is `K / TS`, not `K/(TS·vec)`.** Each iteration advances k by 16 floats (4 vec4s), so the loop runs K/16 times.
2. **GatedDeltaNet `alpha`/`beta` are per-token** (`aRaw[m*16+qk]`, not `aRaw[qk]`). The recurrence is non-stationary; the chunked form needs prefix products `P_m` and weights `w_m = beta_m/P_{m+1}`.
3. **Flash-attention online rescale:** per-element probability must be `exp(score − tile_max)` (not `− m_next`) or `beta` gets applied twice; the sum reduction clobbers the shared array, so per-element probabilities live in a separate `s_p` buffer.
4. **GatedDeltaNet is numerically unstable** with the harness's random weights (`−beta·k·kᵀ` has a large eigenvalue, so the state overflows within ~16 tokens). The `validateGatedDeltaNetGEMM*` functions scale the projection weights by 1/64 (exact in fp16 — exponent shift) so the recurrence stays stable over M=64 and the chunked-vs-sequential comparison is meaningful.
5. **Windows make quirks:** recipes run through `cmd.exe` — `mkdir`/`if exist` paths must use backslashes (`/` is treated as a switch prefix), and shader folder names must not contain spaces (make word-splits `$(wildcard)` output).
6. **KV cache layout is `[token][row]`** (transposed relative to the host `[row][seq]`), so attention shaders index `cache[token*rows + row]`.
7. **Workgroup-wide barriers need every thread present.** For GEMV shaders with a partial last workgroup (N not divisible by 256), out-of-range threads must not early-return — they still run the loop/barriers and carry a `-inf` (`uintBitsToFloat(0xFF800000u)`) accumulator so they can never win the argmax reduction.

---

## 9. Verification Results (M=64, max absolute error vs CPU reference)

| Shader | max_err | GPU time |
|---|---|---|
| GEMM-FP16 / INT8 / INT4 | 0.000080 / 0.000076 / 0.000080 | ~41 ms |
| GEMM-ADD FP16 / INT8 / INT4 | 0.000069 / 0.000069 / 0.000071 | ~14 ms |
| RmsNorm-swiglu-ffn-GEMM FP16 / INT8 / INT4 | 0.0129 / 0.0176 / 0.0142 | 60–77 ms |
| RmsNorm-LinearProj-GEMM FP16 / INT8 / INT4 | 0.00018 / 0.00020 / 0.00021 | 43–51 ms |
| QKV-Rope-GEMM FP16 (q / k-cache / v-cache) | 0.000014 / 0.0021 / 0.062 | ~56 ms |
| QKV-Rope-GEMM INT8 | 0.000014 / 0.0028 / 0.062 | ~53 ms |
| QKV-Rope-GEMM INT4 | 0.000014 / 0.014 / 0.32 | ~51 ms |
| Attention-GEMM FP16 / INT8 / INT4 | 0.000003 / 0.000003 / 0.0017 | ~0.7 ms |
| GatedDeltaNet-GEMM FP16 (out / state S) | 0.000065 / 0.000005 | ~57 ms |
| GatedDeltaNet-GEMM INT8 | 0.000061 / 0.000005 | ~64 ms |
| GatedDeltaNet-GEMM INT4 | 0.000103 / 0.000006 | ~67 ms |
| LMHead-ArgMax FP16 (token index) | exact match | ~117 ms (4096 × 81920 GEMV + reductions) |
| Embed-RmsNorm-LinearProj FP16 (q / k / v / g / a / b) | 0.000164 / 0.000145 / 0.000168 / 0.000175 / 0.000088 / 0.000084 | ~2.6 ms |
| Embed-RmsNorm-LinearProj-GEMM FP16 (q / k / v / g / a / b) | 0.000381 / 0.000313 / 0.000290 / 0.000381 / 0.000198 / 0.000252 | ~155 ms |

The larger k/v-cache errors are fp16/int4 cache quantization, matching the existing GEMV baselines. Note the CPU references are single-threaded and dominate total runtime (`make run` takes several minutes); the GPU shader times above are per-shader query-pool timestamps.

---

## 10. Build & Run

```bash
make clean          # removes bin/ and build/
make                # compiles all shader/**/*.comp -> bin/shader/*.spv, builds bin/main.exe
make run            # = cd bin && main.exe  (runs every validate* and prints max_err + timing)
```
