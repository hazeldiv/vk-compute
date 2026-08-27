# VK Compute

A lightweight Vulkan-based computing engine for matrix operations and LLM
inference (target: Qwen3.5 9B / 32 layers, hybrid INT4/INT8/FP16 layers), built
for AMD Radeon (GCN4-RDNA1-class: RX 580 / RX 6600) through Vulkan compute
shaders.

This branch (`jojo`) includes the changes needed to **build on a modern
MSYS2/MinGW (POSIX) toolchain** and run the engine on a local Radeon GPU via
Vulkan.

## Requirements (Windows)

- **Vulkan SDK** (LunarG). Provides `glslangValidator`, headers, libs.
  e.g. install to `D:\vulkansdk`.
- **MSYS2 (UCRT64)**. Provides `gcc`, `make` (MinGW), shared runtime DLLs.
  e.g. install to `D:\msys2`.
- An **AMD Radeon GPU** with Vulkan support (RX 580 / RX 6600 tested).

## Build

From an **MSYS2 UCRT64** shell, with `VULKAN_SDK` pointed at your SDK:

```bash
export VULKAN_SDK="/d/vulkansdk"           # adjust to your path
export PATH="$VULKAN_SDK/Bin:$PATH"
cd /d/vk-compute

make clean       # removes bin/ and build/
make             # compiles shader/**/*.comp -> bin/shader/*.spv, links bin/main.exe
```

This is the **POSIX Makefile** (no `cmd.exe` syntax). Build a self-contained
binary by copying the MSYS2 runtime DLLs next to the exe:

```bash
# from a UCRT64 shell, once:
for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll \
         libcrypto-3-x64.dll libssl-3-x64.dll libgomp-1.dll; do
  cp "/d/msys2/ucrt64/bin/$d" /d/vk-compute/bin/
done
```

## Run (engine)

By default the engine runs with **synthetic (seeded-random) weights** and emits a
deterministic demo signature. GPU and shaders are exercised for real.

```bash
cd /d/vk-compute
make run                          # = cd bin && main.exe    (engine mode)
bin/main.exe val                  # validation harness (CPU ref vs shaders)
```

Expected on an RX 6600-class GPU: prefill ~85 tok/s, decode ~12 tok/s, zero
errors in stderr.

> Note: `demoWeight/*.bin` are generated/loaded automatically (gitignored via
> `bin/`). Real Qwen weights are NOT shipped in this repo.

## Running a real LLM (Qwen3.5-9B) — pure Vulkan

The engine itself is **100% Vulkan** (C + GLSL compute shaders, no third-party
inference runtime). Everything runs on the Radeon GPU through Vulkan compute
pipelines.

To run the *real* Qwen3.5-9B weights (not the synthetic demo), two components
still need to be built **in this repo, using only Vulkan**:

1. **Safetensors → `demoWeight/*.bin` converter** — loads Qwen's real
   `model.safetensors` shards and writes them in the engine's internal format
   (`transpose_block16`, per-group-256 INT4/INT8/FP16). The `.bin` file format
   is already decoded/verified (see `_setup/bin_probe.py` / `roundtrip.py`).
2. **Tokenizer** — plug Qwen's `tokenizer.json` into the decode loop so output
   ids become text.

No llama.cpp, no GGUF, no external runtime — the weights travel through this
engine's own Vulkan kernels only.

> Status: the `.bin` format is understood and verified (round-trip exact). The
> converter + tokenizer are the next build step in this branch. The syntethic
> demo path already proves the Vulkan pipeline (prefill ~85 tok/s, decode
> ~12 tok/s on RX 6600-class).

## Repository layout

```
Makefile                    # POSIX make (Windows/MSYS2 target)
include/                    # headers: model dims, weights, state, dispatch...
src/                        # Vulkan harness, op compiler, validators, data (fp16/int4/int8)
shader/                     # GLSL compute shaders (FFN, Linear/Full Attention, Utility)
docs/                       # technical summaries & optimization recaps
```

## Gotchas

- Build from the repo root; headers aren't dep-tracked, so `make clean` after
  editing `include/*.h`.
- `core.filemode false` on a filesystem that mangles exec bits (e.g. NTFS).
- The GFX compute kernels prefer small workgroups (TN=32/TS=16) on GCN4.