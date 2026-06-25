# src/tensor.c - Tensor Descriptor Implementation

=============================================================================
FILE: src/tensor.c
PURPOSE: Lightweight tensor view management. Zero-copy descriptors over
         existing Buffer objects. This is NOT a tensor library - it just
         describes how data is laid out in memory.
=============================================================================

## Key Functions

void tensor_create(Tensor* out, Buffer* buffer, TensorDType dtype,
                   int32_t ndim, const int32_t shape[])
  1. Set out->buffer = buffer
  2. Set out->dtype = dtype
  3. Set out->ndim = ndim
  4. Copy shape
  5. Compute row-major strides:
       out->stride[ndim-1] = 1
       for i = ndim-2 down to 0:
           out->stride[i] = out->stride[i+1] * out->shape[i+1]
  6. Set out->offset_bytes = 0
  7. Set name to ""

void tensor_create_with_offset(Tensor* out, Buffer* buffer,
                                uint64_t offset_bytes, TensorDType dtype,
                                int32_t ndim, const int32_t shape[])
  1. tensor_create(out, buffer, dtype, ndim, shape)
  2. out->offset_bytes = offset_bytes

uint64_t tensor_elem_count(const Tensor* t)
  return product of t->shape[0..t->ndim-1]

uint64_t tensor_num_bytes(const Tensor* t)
  return tensor_elem_count(t) * TENSOR_DTYPE_SIZE(t->dtype)

uint64_t tensor_buffer_offset(const Tensor* t)
  return t->offset_bytes + tensor_elem_count(t) * TENSOR_DTYPE_SIZE(t->dtype)
  // Note: used for bounds checking

bool tensor_is_contiguous(const Tensor* t)
  - Check if stride matches row-major layout
  - stride[i] == stride[i+1] * shape[i+1] for all i except last

Tensor tensor_reshape(const Tensor* src, int32_t ndim, const int32_t shape[])
  1. Verify total element count matches
  2. Create new tensor with same buffer and offset but new shape
  3. Compute new strides (row-major assumed)
  4. Return new Tensor (NOT a view - caller owns the new Tensor struct)

TensorView tensor_view(const Tensor* src, TensorLayout layout)
  1. Create TensorView wrapping src
  2. Apply permutation: new_stride[layout.perm[i]] = old_stride[i]
  3. Swap shape accordingly: new_shape[layout.perm[i]] = old_shape[i]
  4. Set all slice_start[i] = 0, slice_end[i] = -1

TensorView tensor_slice(const Tensor* src, int32_t dim,
                        int32_t start, int32_t end)
  1. Create TensorView wrapping src
  2. Set slice_start[dim] = start
  3. Set slice_end[dim] = (end == -1) ? src->shape[dim] : end
  4. Adjust the base tensor's shape and offset:
       new_shape[dim] = slice_end - slice_start
       new_offset += slice_start * old_stride[dim] * dtype_size

void tensor_layout_BHSD(int32_t B, int32_t H, int32_t S, int32_t D,
                        int32_t strides[4])
  // BHSD = [Batch, Heads, Seq, HeadDim]
  // Memory layout: fastest varying is D (stride=1), then S (stride=D),
  // then H (stride=S*D), then B (stride=H*S*D)
  strides[0] = H * S * D;  // B dimension
  strides[1] = S * D;       // H dimension
  strides[2] = D;           // S dimension
  strides[3] = 1;           // D dimension

void tensor_layout_BSHD(int32_t B, int32_t S, int32_t H, int32_t D,
                        int32_t strides[4])
  // BSHD = [Batch, Seq, Heads, HeadDim]  (PyTorch default)
  strides[0] = S * H * D;
  strides[1] = H * D;
  strides[2] = D;
  strides[3] = 1;

void tensor_print(const Tensor* t)
  - Print: name, dtype, shape as [d0 x d1 x d2 x ...], offset, strides
  - Example: "hidden_states: f16 [1 x 128 x 3584] offset=0 stride=[458752,3584,1]"

## Common Tensor Layouts Used in LLM Inference

Layout: BHSD  [Batch, Heads, Seq, HeadDim]
  - Best for Flash Attention (contiguous heads, then sequence)
  - K, V cache naturally fits this layout
  - Query: after splitting from QKV, already in BHSD

Layout: BSHD  [Batch, Seq, Heads, HeadDim]
  - Standard PyTorch attention format
  - Easier to think about: [seq, hidden] then split heads
  - Used for embeddings, MLP hidden states

## Tensor Naming Conventions (for debugging)

  "embeddings"       - token embeddings [vocab, hidden]
  "hidden_states"    - [batch, seq, hidden] current activations
  "qkv"              - concatenated QKV [batch, seq, (q+kv+kv)*dim]
  "q"                - queries [batch, num_heads, seq, head_dim]
  "k"                - keys [batch, num_kv_heads, seq, head_dim]
  "v"                - values [batch, num_kv_heads, seq, head_dim]
  "attn_scores"       - [batch, num_heads, seq, seq] raw attention logits
  "attn_weights"      - [batch, num_heads, seq, seq] softmax output
  "attn_output"       - [batch, num_heads, seq, head_dim] after O projection
  "mlp_gate"          - [batch, seq, intermediate] gate projection output
  "mlp_up"            - [batch, seq, intermediate] up projection output
  "mlp_output"        - [batch, seq, hidden] final MLP output
  "logits"            - [batch, seq, vocab] final logits
  "k_cache"           - [layer, batch, num_kv_heads, max_seq, head_dim]
  "v_cache"           - [layer, batch, num_kv_heads, max_seq, head_dim]
