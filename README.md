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

## Running a real LLM (Qwen3.5-9B) via Vulkan + llama.cpp

For *real* Qwen text output (tokenizer + real weights), use `llama.cpp` built
with the **Vulkan backend** and a GGUF of Qwen3.5-9B. This is the production
path (vk-compute itself is a research engine and does not yet provide a
safetensors->internal converter / tokenizer).

```bash
# 1. Build llama.cpp with Vulkan (MSYS2 UCRT64 + Vulkan SDK in path):
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DGGML_CUDA=OFF -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target llama-cli llama-server -j 8

# 2. Copy MSYS2 runtime DLLs next to the exe (same list as above).

# 3. Download a GGUF:
#    huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf

# 4. Chat interactively (all layers on GPU):
D:\llama.cpp\build-vulkan\bin\llama-cli.exe -m D:\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf -c 4096 -ngl 99

# 5. If "OutOfDeviceMemory" on an 8 GB card, shrink the KV cache:
llama-cli.exe -m ...\Qwen3.5-9B-Q4_K_M.gguf -c 4096 -ngl 99
```

Notes:
- `-ngl 99` offloads every layer to Vulkan/GPU. Dropping it falls back to CPU.
- `-c 4096` keeps the KV cache small enough to fit 5.6 GB model + cache in ~8 GB VRAM.
- Simpler/faster: just `ollama run qwen3.5:9b` if you prefer an one-liner.

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