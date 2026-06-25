# LLM Architecture Blueprint
# ============================
#
# This is the master reference for file organization.
# See individual docs/00_* files for full details of each module.

## FILE OVERVIEW

include/model.h
    ModelConfig struct
    LayerWeights struct
    ModelState struct
    model_load()
    model_unload()

include/tensor.h
    Tensor struct
    TensorView struct
    tensor_create()
    tensor_view()
    tensor_reshape()

include/operator.h
    OpRegistry struct
    OpRegistryEntry struct
    op_register()
    op_get()
    op_layernorm()
    op_rmsnorm()
    op_rope()
    op_attention()
    op_attention_flash()
    op_mlp()
    op_softmax()
    op_quantize()
    op_dequantize()

include/inference.h
    InferenceContext struct
    inference_context_create()
    inference_context_destroy()
    inference_forward()
    inference_generate()
    inference_reset_kv_cache()

include/tokenizer.h
    Tokenizer struct
    tokenizer_load()
    tokenizer_encode()
    tokenizer_decode()
    tokenizer_free()

include/engine.h
    LLMEngine struct
    llm_engine_create()
    llm_engine_destroy()
    llm_engine_load_model()
    llm_engine_generate()

src/model.c
    Implementation of model loading.

src/tensor.c
    Implementation of tensor management.

src/operator.c
    Operator registry + CPU-side wrappers.

src/operators/layernorm.c
    GPU LayerNorm kernel.

src/operators/rmsnorm.c
    GPU RMSNorm kernel.

src/operators/rope.c
    GPU Rotary Position Embedding kernel.

src/operators/attention.c
    GPU naive attention kernel.

src/operators/attention_flash.c
    GPU Flash Attention kernel.

src/operators/mlp.c
    GPU MLP/FeedForward kernel.

src/operators/silu.c
    GPU SiLU activation kernel.

src/operators/softmax.c
    GPU Softmax kernel.

src/operators/quantize.c
    GPU quantization kernels.

src/inference.c
    Implementation of forward pass and generation.

src/tokenizer.c
    Tokenizer implementation.

src/engine.c
    Top-level facade implementation.
