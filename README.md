# VK Compute

A Vulkan-based LLM inference engine, written from scratch in C and GLSL compute shaders. It runs **Qwen3.5-9B** locally on an **AMD RX 580 8 GB** — a GPU with no AI acceleration — at ~25 tokens per second, with no ML framework involved: no PyTorch, no CUDA, no llama.cpp.

The engine implements the model's full text stack: 32 transformer layers with hybrid attention (gated delta-net and full attention), per-layer FP16/INT8/INT4 quantization to fit 8 GB of VRAM, chunked prefill, look-ahead decode, and an on-GPU sampler (temperature, top-k, top-p, min-p, repetition penalty). Output is verified against the HuggingFace reference: prefill layers track it at 0.96–0.9999 cosine correlation, and greedy decode matches it token-for-token.

To fit the vocabulary in VRAM, the model's 248,320-token head is pruned to 86,016 rows. A companion toolchain (`tools/pruner`) keeps that prune loss-free: it detects which tokens the model actually wants to use — by replaying generated sequences against the full lm_head on CPU — and protects them through every re-prune.

## Project Layout

| Path | Contents |
|---|---|
| `src/`, `include/` | C engine: weight loading, op dispatch, token server |
| `shader/` | 121 GLSL compute shaders (GEMV/GEMM, attention, sampler) |
| `vk_llm.py` | Python frontend: chat template, tokenization, streaming, sampling config |
| `tools/pruner/` | Vocab pruning toolchain and verification harness |
| `docs/VK-COMPUTE-SUMMARY.md` | Full technical documentation |
| `model/` | Weights and vocab artifacts (not in git) |

## Requirements

- Windows with MinGW-w64 / MSYS2 UCRT64 (`gcc`, `make`)
- Vulkan SDK (tested with 1.4.350)
- Python 3.9+ with [uv](https://docs.astral.sh/uv/), plus `tokenizers`
- A GPU with 8 GB VRAM (built and tuned for the RX 580)
- Qwen3.5-9B safetensors under `model/Qwen3.5-9B-weight/`, with the pruned `lm_head.*` / `embed_tokens.*` files alongside and the pruned tokenizer in `model/Qwen3.5-pruned-vocab/`

## Build & Run

```bash
# Python deps (once)
uv venv .venv
uv pip install --python .venv/Scripts/python.exe tokenizers

# Build from the repo root
make clean
make

# Validate every shader against its CPU reference
cd bin && main.exe val

# Generate
.venv/Scripts/python.exe vk_llm.py model/Qwen3.5-9B-weight 4096 model/Qwen3.5-pruned-vocab "your prompt here" [thinking]
```

Sampling behavior is set by the constants at the top of `vk_llm.py` (`is_sampling`, `temperature`, `top_k`, `top_p`, `min_p`, `rep_penalty`, `penalty_len`, `seed`).

## Documentation

`docs/VK-COMPUTE-SUMMARY.md` covers the architecture in depth: every shader's algorithm and memory layout, the server wire protocol, the VRAM budget, the validation harness, and the engineering notes behind the quantization, decode, and vocab-pruning decisions.
