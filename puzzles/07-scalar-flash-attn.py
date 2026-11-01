"""
Puzzle 07: Scalar FlashAttention
==============
From softmax to FlashAttention, we just need some computation.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Now we have conquered softmax / online softmax, we can now implement one of the most important
operator in LLMs: FlashAttention.

To ensure a progressive learning experience, we will implement a scalar version of FlashAttention.
And we also remove the multi-head attention part. So in total we only have two dimensions: batch
size B and sequence length S, which are aligned with N, M in the previous puzzle. After such
simplification, you will find we are not so far from the FlashAttention algorithm. And with
TileLang, we can easily extend it to the full FlashAttention.

06-1: Simplified Scalar Flash Attention.

Inputs:
    Q: Tensor([B, S], float32)  # input tensor
    K: Tensor([B, S], float32)  # input tensor
    V: Tensor([B, S], float32)  # input tensor
    B: int   # batch size dimension. 1 <= B <= 256
    S: int   # sequence length dimension. 1 <= S <= 16384

Output:
    O: Tensor([B, S], float32)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row
    QK: Tensor([B, S], float32)  # results of q*k
    P:  Tensor([B, S], float32)  # results of softmax(q*k) (not divided by summation).

Definition:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""


def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
    assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    return torch.softmax(Q * K, dim=1).mul_(V)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    # TODO: Implement this function
    with T.Kernel(B, threads=512) as (bx):
        Q_row = T.alloc_fragment((1, S), dtype)
        QK_row = T.alloc_fragment((1, S), dtype)

        T.copy(Q[bx, 0], Q_row)
        T.copy(K[bx, 0], QK_row)
        for i in T.Parallel(S):
            QK_row[0, i] = Q_row[0, i] * QK_row[0, i]
        QK_row_max = T.alloc_fragment((1,), dtype)

        T.reduce_max(QK_row, QK_row_max, dim=1)
        for i in T.Parallel(S):
            QK_row[0, i] = T.exp2(log2_e * (QK_row[0, i] - QK_row_max[0]))

        QK_row_sum = T.alloc_fragment((1,), dtype)

        T.reduce_sum(QK_row, QK_row_sum, dim=1)
        for i in T.Parallel(S):
            O[bx, i] = QK_row[0, i] / QK_row_sum[0] * V[bx, i]

    # with T.Kernel(B // BLOCK_B, threads=256) as (bx):
    #     Q_tile = T.alloc_shared((BLOCK_B, BLOCK_S), dtype)
    #     QK_tile = T.alloc_shared((BLOCK_B, BLOCK_S), dtype)

    #     QK_tile_max = T.alloc_shared((BLOCK_B,), dtype)
    #     QK_new_max = T.alloc_shared((BLOCK_B,), dtype)
    #     QK_tile_sum = T.alloc_shared((BLOCK_B,), dtype)

    #     QK_row_max = T.alloc_shared((BLOCK_B,), dtype)
    #     QK_row_sum = T.alloc_shared((BLOCK_B,), dtype)

    #     T.fill(QK_row_max, -T.infinity(dtype))
    #     T.fill(QK_row_sum, T.float32(0))
    #     for k in T.serial(S // BLOCK_S):
    #         T.copy(Q[bx * BLOCK_B, k * BLOCK_S], Q_tile)
    #         T.copy(K[bx * BLOCK_B, k * BLOCK_S], QK_tile)
    #         for i, j in T.Parallel(BLOCK_B, BLOCK_S):
    #             QK_tile[i, j] = Q_tile[i, j] * QK_tile[i, j]
    #         T.copy(QK_tile, O[bx * BLOCK_B, k * BLOCK_S])
    #         T.reduce_max(QK_tile, QK_tile_max, dim=1)
    #         for i, j in T.Parallel(BLOCK_B, BLOCK_S):
    #             QK_tile[i, j] = T.exp2(log2_e * (QK_tile[i, j] - QK_tile_max[i]))
    #         T.reduce_sum(QK_tile, QK_tile_sum, dim=1)

    #         for i in T.Parallel(BLOCK_B):
    #             QK_new_max[i] = T.max(QK_tile_max[i], QK_row_max[i])
    #             QK_row_sum[i] = QK_row_sum[i] * T.exp2(
    #                 log2_e * (QK_row_max[i] - QK_new_max[i])
    #             ) + QK_tile_sum[i] * T.exp2(log2_e * (QK_tile_max[i] - QK_new_max[i]))
    #             QK_row_max[i] = QK_new_max[i]

    #     # now we have row_max, row_sum, next, we can compute the final result
    #     for i, j in T.Parallel(BLOCK_B, S):
    #         O[bx * BLOCK_B + i, j] = (
    #             T.exp2(log2_e * (O[bx * BLOCK_B + i, j] - QK_row_max[i]))
    #             * V[bx * BLOCK_B + i, j]
    #             / QK_row_sum[i]
    #         )

    return O


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()
