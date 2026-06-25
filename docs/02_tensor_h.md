# include/tensor.h - Lightweight Tensor Descriptor

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "buffer.h"

// ---------------------------------------------------------------------------
// DataType mapping
// ---------------------------------------------------------------------------

typedef enum {
    TENSOR_F32,
    TENSOR_F16,
    TENSOR_BF16,
    TENSOR_I8,
    TENSOR_I4,
    TENSOR_U8,
} TensorDType;

// ---------------------------------------------------------------------------
// Tensor - a descriptor referring to a buffer region
// ---------------------------------------------------------------------------
// A Tensor is a lightweight VIEW over a Buffer (or part of one).
// It does NOT own memory. It describes: shape, stride, offset, dtype.
// This mirrors how GGML and PyTorch handle tensor descriptors.
// ---------------------------------------------------------------------------

typedef struct {
    // The underlying buffer (reference, NOT owned)
    Buffer* buffer;

    // Byte offset into the buffer where tensor data starts
    uint64_t offset_bytes;

    // Shape: up to 4 dimensions. -1 means dynamic (inferred).
    int32_t shape[4];
    int32_t ndim;  // Number of valid dimensions (1 to 4)

    // Stride in elements (not bytes) for each dimension.
    // For a contiguous tensor: stride[i] = prod(shape[i+1:])
    // shape = [B, H, S, D], ndim=4:
    //   stride[0] = H * S * D   (B dimension stride)
    //   stride[1] = S * D        (H dimension stride)
    //   stride[2] = D            (S dimension stride)
    //   stride[3] = 1            (D dimension stride)
    int32_t stride[4];

    // Element data type
    TensorDType dtype;

    // Optional name for debugging
    char name[32];
} Tensor;

// Convenience macro to compute element size in bytes
#define TENSOR_DTYPE_SIZE(dt) \
    ((dt) == TENSOR_F32 ? 4 : \
     (dt) == TENSOR_F16 || (dt) == TENSOR_BF16 ? 2 : \
     (dt) == TENSOR_I8 || (dt) == TENSOR_U8 ? 1 : 1)

// ---------------------------------------------------------------------------
// TensorView - a transposed or sliced view
// ---------------------------------------------------------------------------

typedef struct {
    Tensor base;
    // Slice ranges: for a view, these specify [start, end) per dimension
    // A view does NOT copy data, just changes how it's accessed.
    int32_t slice_start[4];
    int32_t slice_end[4];   // -1 means end of dimension
} TensorView;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * tensor_create()
 * Creates a tensor descriptor wrapping an existing Buffer.
 * Does NOT allocate memory.
 *
 * Example - creating a [B, H, S, D] tensor:
 *   Tensor t;
 *   tensor_create(&t, buffer, TENSOR_F16, 4,
 *                 (int32_t[]){batch, num_heads, seq_len, head_dim});
 */
void tensor_create(
    Tensor* out,
    Buffer* buffer,
    TensorDType dtype,
    int32_t ndim,
    const int32_t shape[]
);

/**
 * tensor_create_with_offset()
 * Like tensor_create but with explicit byte offset.
 */
void tensor_create_with_offset(
    Tensor* out,
    Buffer* buffer,
    uint64_t offset_bytes,
    TensorDType dtype,
    int32_t ndim,
    const int32_t shape[]
);

/**
 * tensor_view()
 * Creates a view of a tensor with different strides (e.g., transposed).
 * The view refers to the SAME buffer memory.
 *
 * Common use: transpose last two dims (for attention):
 *   input shape [B, H, S, D] -> want [B, H, D, S]
 *   TensorView v = tensor_view(&t, perm);
 */
typedef struct {
    int32_t perm[4];  // Dimension permutation, e.g., {0, 1, 3, 2}
} TensorLayout;

TensorView tensor_view(const Tensor* src, TensorLayout layout);

/**
 * tensor_reshape()
 * Returns a new tensor descriptor with different shape but SAME strides.
 * Only valid if total element count matches.
 */
Tensor tensor_reshape(const Tensor* src, int32_t ndim, const int32_t shape[]);

/**
 * tensor_slice()
 * Returns a view with sliced dimensions [start:end].
 */
TensorView tensor_slice(
    const Tensor* src,
    int32_t dim,
    int32_t start,
    int32_t end
);

/**
 * tensor_elem_count()
 * Returns total number of elements (product of shape).
 */
uint64_t tensor_elem_count(const Tensor* t);

/**
 * tensor_num_bytes()
 * Returns total bytes for this tensor.
 */
uint64_t tensor_num_bytes(const Tensor* t);

/**
 * tensor_print()
 * Debug print: name, shape, dtype, stride.
 */
void tensor_print(const Tensor* t);

// ---------------------------------------------------------------------------
// Common layout helpers
// ---------------------------------------------------------------------------

// Attention-friendly layouts
// BHSD: [batch, heads, seq, head_dim]     <- Best for Flash Attention
// BSHD: [batch, seq, heads, head_dim]     <- Standard PyTorch
// SBHD: [seq, batch, heads, head_dim]     <- Used by some older implementations

/**
 * tensor_layout_BHSD()
 * Returns the strides for a BHSD layout tensor.
 * B=0, H=1, S=2, D=3, row-major strides: {H*S*D, S*D, D, 1}
 */
void tensor_layout_BHSD(int32_t B, int32_t H, int32_t S, int32_t D,
                        int32_t strides[4]);

/**
 * tensor_layout_BSHD()
 * Returns the strides for a BSHD layout tensor.
 * B=0, S=1, H=2, D=3, row-major strides: {S*H*D, H*D, D, 1}
 */
void tensor_layout_BSHD(int32_t B, int32_t S, int32_t H, int32_t D,
                        int32_t strides[4]);
