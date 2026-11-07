"""
Puzzle 09: Convolution
==============
Convolution is another fundamental computation pattern in deep learning operators.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Convolution uses a sliding window approach to compute over a input tensors. The main
characteristics of convolution is that it has strong data reuse patterns and requires careful
memory access optimization. But with TileLang, we can ignore most of these details and focus on
the logic.

In this puzzle, we remove the "channel (C)" dimension to simplify the problem. We first look at
the 1D convolution case, then extend to 2D. And we will learn how to use shared memory of GPU in
this chapter.

08-1: 1D Convolution.

Inputs:
    X: Tensor([N, L], float16)  # input tensor
    K: Tensor([KL,], float16)  # kernel tensor
    N: int   # batch size dimension. 1 <= N <= 64
    L: int   # length dimension. 1 <= L <= 1024
    KL: int  # kernel height. 1 <= KH <= 32

Output:
    O: Tensor([N, L], float16)  # output tensor

Intermediates:
    ACC: float32  # accumulator

Definition:
    for i in range(N):
        for j in range(L):
            ACC = 0
            for k in range(KL):
                if j + k < L:  # boundary check
                    ACC += X[i, j + k] * K[k]
            O[i, j] = ACC
"""


"""
We can first consider a naive implementation. We can parallelize the outer loop over `N` and `L` to
different blocks with `T.Kernel`.
For the loop iterating `BLOCK_L`, we can use a serial implementation for now. Be careful that the
data dependency in the convolution.
"""


def ref_conv1d(X: torch.Tensor, K: torch.Tensor):
    assert len(X.shape) == 2
    assert len(K.shape) == 1
    assert X.dtype == K.dtype == torch.float16

    # for i in range(N):
    #     for j in range(L):
    #         O[i, j] = 0
    #         for k in range(KL):
    #             if j + k < L:  # boundary check
    #                 O[i, j] += X[i, j + k] * K[k]

    N, L = X.shape
    KL = K.shape[0]

    padding_size = KL - 1
    X_padded = torch.nn.functional.pad(X.view(N, 1, L), (0, padding_size))

    return torch.conv1d(
        input=X_padded,
        weight=K.view(1, 1, KL),
    ).view(N, L)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_naive(X, K, BLOCK_N: int, BLOCK_L: int):
    N, L, KL = T.const("N, L, KL")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, L), dtype)
    K: T.Tensor((KL,), dtype)
    O = T.empty((N, L), dtype)

    # TODO: Implement this function
    with T.Kernel(N, threads=64) as (bx):
        X_row = T.alloc_shared((L,), accum_dtype)
        X_reduce = T.alloc_fragment((L,), accum_dtype)
        K_local = T.alloc_shared((KL,), accum_dtype)
        T.copy(X[bx, 0], X_row)
        T.clear(X_reduce)
        T.copy(K, K_local)
        for i in T.serial(KL):
            for j in T.Parallel(L):
                if j + i < L:
                    X_reduce[j] += X_row[j + i] * K_local[i]
        T.copy(X_reduce, O[bx, 0])
    return O


def run_conv1d_naive():
    print("\n=== Convolution 1D Naive ===\n")
    N = 128
    L = 128
    BLOCK_N = 16
    BLOCK_L = 32
    KL = 32
    test_puzzle(
        tl_conv1d_naive,
        ref_conv1d,
        {"N": N, "L": L, "KL": KL, "BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L},
    )


"""
The naive implementation of Conv 1D works but it is not efficient. Remember that we mentioned
Tensor Core and `T.gemm` in previous puzzle? Actually, we can also convert the convolution problem
to a GEMM problem through a transformation called `im2col`. The idea is to transform the
convolution into a matrix multiplication where each row of the input matrix corresponds to a local
patch of the input tensor, and the kernel is reshaped into a matrix. This allows us to leverage
highly optimized GEMM implementations.

To present GEMM from degenerating to GEMV, we need to introduce an output channel dimension F.

08-2: 1D Convolution with multiple output channels.

Inputs:
    X: Tensor([N, L], float16)  # input tensor
    K: Tensor([KL, F], float16)  # kernel tensor
    N: int   # batch size dimension. 1 <= N <= 64
    L: int   # length dimension. 1 <= L <= 1024
    KL: int  # kernel height. 1 <= KH <= 32
    F: int   # filter channels. 32 <= F <= 128

Output:
    O: Tensor([N, L, F], float16)  # output tensor

Intermediates:
    ACC: float32  # accumulator

Definition:
    for i in range(N):
        for j in range(L):
            for f in range(F):
                ACC = 0
                for k in range(KL):
                    if j + k < L:  # boundary check
                        ACC += X[i, j + k] * K[k, f]
                O[i, j, f] = ACC
"""


def ref_conv1d_multi_outchannel(X: torch.Tensor, K: torch.Tensor):
    assert len(X.shape) == 2
    assert len(K.shape) == 2
    assert X.dtype == K.dtype == torch.float16

    # for i in range(N):
    #     for j in range(L):
    #         for f in range(F):
    #             O[i, j, f] = 0
    #             for k in range(KL):
    #                 if j + k < L:  # boundary check
    #                     O[i, j, f] += X[i, j + k] * K[k, f]

    N, L = X.shape
    KL, F = K.shape

    padding_size = KL - 1
    X_padded = torch.nn.functional.pad(X.view(N, 1, L), (0, padding_size))

    return (
        torch.conv1d(
            input=X_padded,
            weight=K.permute(1, 0).view(F, 1, KL),
        )
        .permute(0, 2, 1)
        .contiguous()
    )


"""
First let's implement the trivial extension of conv1d to multiple output channels, based on the
above F=1 version.
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_multi_outchannel(X, K, BLOCK_N: int, BLOCK_L: int):
    N, L, KL, F = T.const("N, L, KL, F")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, L), dtype)
    K: T.Tensor((KL, F), dtype)
    O = T.empty((N, L, F), dtype)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, L // BLOCK_L, threads=32) as (bx, by):
        X_tile = T.alloc_shared((BLOCK_N, BLOCK_L + KL - 1), dtype)
        K_tile = T.alloc_shared((KL, F), dtype)
        XK_reduce = T.alloc_fragment((F), accum_dtype)
        T.copy(X[bx * BLOCK_N, by * BLOCK_L], X_tile)
        T.copy(K, K_tile)
        for i in T.serial(BLOCK_L):
            for j in T.serial(BLOCK_N):
                T.clear(XK_reduce)
                for k in T.serial(KL // 4):
                    for f in T.Parallel(F):
                        XK_reduce[f] += (
                            T.cast(X_tile[j, i + k * 4 + 0], accum_dtype)
                            * T.cast(K_tile[k * 4 + 0, f], accum_dtype)
                            + T.cast(X_tile[j, i + k * 4 + 1], accum_dtype)
                            * T.cast(K_tile[k * 4 + 1, f], accum_dtype)
                            + T.cast(X_tile[j, i + k * 4 + 2], accum_dtype)
                            * T.cast(K_tile[k * 4 + 2, f], accum_dtype)
                            + T.cast(X_tile[j, i + k * 4 + 3], accum_dtype)
                            * T.cast(K_tile[k * 4 + 3, f], accum_dtype)
                        )
                for f in T.Parallel(F):
                    O[bx * BLOCK_N + j, by * BLOCK_L + i, f] = XK_reduce[f]
    return O


"""
Then let's try im2col and use T.gemm to speedup the computation.
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_im2col(X, K, BLOCK_N: int, BLOCK_L: int):
    N, L, KL, F = T.const("N, L, KL, F")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, L), dtype)
    K: T.Tensor((KL, F), dtype)
    O = T.empty((N, L, F), dtype)

    # TODO: Implement this function
    # with T.Kernel(N // BLOCK_N, L // BLOCK_L, threads=32) as (bx, by):
    #     X_tile = T.alloc_shared((BLOCK_N, BLOCK_L + KL - 1), accum_dtype)
    #     X_matrix = T.alloc_shared(
    #         (
    #             BLOCK_L,
    #             BLOCK_N,
    #             KL,
    #         ),
    #         accum_dtype,
    #     )
    #     K_tile = T.alloc_shared((KL, F), accum_dtype)
    #     XK_reduce = T.alloc_fragment((BLOCK_N, F), accum_dtype)
    #     T.copy(X[bx * BLOCK_N, by * BLOCK_L], X_tile)
    #     T.copy(K, K_tile)
    #     for i in T.Parallel(BLOCK_L):
    #         T.copy(X_tile[0:BLOCK_N, i : (i + KL)], X_matrix[i, 0:BLOCK_N, 0:KL])
    #     T.gemm(X_matrix, K_tile, XK_reduce, clear_accum=True)
    #     for i in T.Parallel(BLOCK_L):
    #         T.copy(
    #             XK_reduce,
    #             O[
    #                 (bx * BLOCK_N) : (bx * BLOCK_N + BLOCK_N),
    #                 (by * BLOCK_L + i),
    #                 0:F,
    #             ],
    #         )
    BLOCK_M = BLOCK_N * BLOCK_L

    with T.Kernel(
        T.ceildiv(N, BLOCK_N),
        T.ceildiv(L, BLOCK_L),
        threads=256,
    ) as (bx, by):
        A_col = T.alloc_shared((BLOCK_M, KL), dtype)
        K_tile = T.alloc_shared((KL, F), dtype)
        C_tile = T.alloc_fragment((BLOCK_M, F), accum_dtype)

        for m, k in T.Parallel(BLOCK_M, KL):
            j = m // BLOCK_L
            i = m % BLOCK_L
            n = bx * BLOCK_N + j
            pos = by * BLOCK_L + i + k
            A_col[m, k] = T.if_then_else(
                (n < N) & (pos < L),
                X[n, pos],
                0,
            )
        T.copy(K, K_tile)
        T.gemm(
            A_col,
            K_tile,
            C_tile,
            clear_accum=True,
        )
        for m, f in T.Parallel(BLOCK_M, F):
            j = m // BLOCK_L
            i = m % BLOCK_L

            n = bx * BLOCK_N + j
            pos = by * BLOCK_L + i

            if (n < N) & (pos < L):
                O[n, pos, f] = C_tile[m, f]
    return O


def run_conv1d_im2col():
    print("\n=== Convolution 1D im2col ===\n")
    N = 128
    L = 128
    BLOCK_N = 16
    BLOCK_L = 32
    KL = 32
    F = 32
    args_dict = {
        "N": N,
        "L": L,
        "KL": KL,
        "F": F,
        "BLOCK_N": BLOCK_N,
        "BLOCK_L": BLOCK_L,
    }
    test_puzzle(tl_conv1d_multi_outchannel, ref_conv1d_multi_outchannel, args_dict)
    test_puzzle(tl_conv1d_im2col, ref_conv1d_multi_outchannel, args_dict)
    bench_puzzle(
        tl_conv1d_multi_outchannel,
        ref_conv1d_multi_outchannel,
        args_dict,
        bench_torch=True,
        bench_name="Conv1D Multi OutChannel Naive",
    )
    bench_puzzle(
        tl_conv1d_im2col,
        ref_conv1d_multi_outchannel,
        args_dict,
        bench_torch=False,
        bench_name="Conv1D im2col",
    )


if __name__ == "__main__":
    run_conv1d_naive()
    run_conv1d_im2col()
