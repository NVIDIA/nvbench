# Copyright 2025 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 with the LLVM exception
#  (the "License"); you may not use this file except in compliance with
#  the License.
#
#  You may obtain a copy of the License at
#
#      http://llvm.org/foundation/relicensing/LICENSE.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import sys
from typing import Tuple

import cuda.bench as bench
import cuda.bindings.driver as driver
import cuda.core as core
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import numpy as np
from cutlass.cute.runtime import from_dlpack


def as_bindings_Stream(cs: bench.CudaStream) -> driver.CUstream:
    return driver.CUstream(cs.addressof())


def as_core_Stream(cs: bench.CudaStream) -> core.Stream:
    return core.Stream.from_handle(cs.addressof())


def make_tensor(arr_h: np.ndarray, dev_buf: core.Buffer, dev_id: int | None):
    cp_memview = cp.cuda.UnownedMemory(
        int(dev_buf.handle), dev_buf.size, dev_buf, -1 if dev_id is None else dev_id
    )
    zero_offset = 0
    return from_dlpack(
        cp.ndarray(
            arr_h.shape,
            dtype=arr_h.dtype,
            memptr=cp.cuda.MemoryPointer(cp_memview, zero_offset),
        ),
        assumed_align=16,
    )


class SGemm:
    """
    Adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/sgemm.py
    """

    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 8),
        num_stages: int = 3,
        num_threads: int = 256,
    ):
        self._cta_tiler = cta_tiler
        self._num_stages = num_stages
        self._num_threads = num_threads
        assert num_threads > 0, "needs at least one thread"
        assert num_threads % 16 == 0, "multiples of 16 required for MMA thread layout"

        self._bM, self._bN, self._bK = self._cta_tiler
        assert self._bM % 16 == 0, "multiple of 16 required for tile dimension M"
        assert self._bN % 16 == 0, "multiple of 16 required for tile dimension N"
        assert self._num_stages >= 3, "num_stages must be greater than or equal to 3"
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        stream: driver.CUstream = driver.CUstream(
            driver.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        # ///////////////////////////////////////////////////////////////////////////////
        # Create layouts for shared memory for A and B:
        #   - sA/sB is m/n-major to vectorized copies from shared
        #       memory to registers. This is because the MMA layouts
        #       for sA/sB are also m/n-major
        #   - When gA/gB is k-major, pad 4 elements to reduce bank conflicts
        # ///////////////////////////////////////////////////////////////////////////////

        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Create copy layouts that will be used for asynchronous
        # global memory -> shared memory copies:
        #   - The majorness of tA/tB follows the majorness of gA/gB
        #   - For k-major, these layouts will copy values one-by-one from
        #       from global memory, without vectorizing
        #   - For m/n-major, it will vectorize to a 128bit copy for faster
        #       data transfer between global and shared memory, as long
        #       as the alignment of the tensor allows it. Otherwise, it
        #       defaults to a non-vectorized copy
        # ///////////////////////////////////////////////////////////////////////////////

        tA = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        tB = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        vA = cute.make_layout((1, 1))
        vB = cute.make_layout((1, 1))
        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mB.element_type.width,
        )
        if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized = 4 if (mA.layout[0].max_alignment % 16 == 0) else 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            major_mode_size = self._bM // num_vectorized
            tA = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vA = cute.make_layout((num_vectorized, 1))

        if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized = 4 if (mB.layout[0].max_alignment % 16 == 0) else 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            major_mode_size = self._bN // num_vectorized
            tB = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vB = cute.make_layout((num_vectorized, 1))

        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

        # ///////////////////////////////////////////////////////////////////////////////
        # Create layouts for GEMM:
        # We tile an MMA atom across a tensor. `atoms_layout` is the layout
        # of atoms in the tiled MMA. (Because we use an `MmaUniversalOp`,
        # which has a trivial 1x1x1 MMA trait, `atoms_layout` is also
        # simply the thread layout for C.) `permutation_tiler` reorders the
        # elements of the tensor that the tiled MMA is applied to.
        # Different combinations of `atoms_layout` and `permutation_tiler`
        # values can create different MMA thread-value patterns.
        #
        # Here, the MMA layout is set so that each thread copies four
        # consecutive elements from shared memory to registers.
        # `permutation_tiler_M/N` maps the elements handled by each thread
        # to the permuted element in the tensor.
        # For increasing indices in the tensor, the thread ID that reads it is:
        #   - (without permutation) ==>
        #      0 1 2 ... 15 0 1 2 ... 15 0 1 2 ... 15 0 1 2 ... 15 ......
        #   - (with permutation) ==>
        #      0 0 0 0 1 1 1 1 2 2 2 2 ... 15 15 15 15 0 0 0 0 1 1 1 1 ......
        # ///////////////////////////////////////////////////////////////////////////////
        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        permutation_tiler_M = cute.make_layout(
            (atoms_layout.shape[0], 4), stride=(4, 1)
        )
        permutation_tiler_N = cute.make_layout(
            (atoms_layout.shape[1], 4), stride=(4, 1)
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            atoms_layout,
            permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
        )

        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            epilogue_op,
        ).launch(
            grid=grid_dim,
            block=[cute.size(atoms_layout), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Thread and block indices
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # gA: (BLK_M, BLK_K, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
        # ///////////////////////////////////////////////////////////////////////////////
        gA = cute.local_tile(
            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gB = cute.local_tile(
            mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )

        # Move the pointer of gA/gB in the `-k`` direction, making the first
        # tile (instead of the last one) irregular in shape when k is irregular.
        # We first handle the irregular tile to avoid checking for this
        # condition within the mainloop.
        residue_k = mA.shape[1] - self._bK * gA.shape[2]
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread.
        # sA:   (BLK_M, BLK_K, PIPE)       , sB:   (BLK_N, BLK_K, PIPE)
        # tAgA: (CPY, CPY_M, CPY_K, k)     , tBgB: (CPY, CPY_N, CPY_K, k)
        # tAsA: (CPY, CPY_M, CPY_K, PIPE)  , tBsB: (CPY, CPY_N, CPY_K, PIPE)
        # ///////////////////////////////////////////////////////////////////////////////
        # Create shared memory buffer
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when the problem shape
        # isn't a multiple of the tile shape. If tApA/B[i] is 0, then do not
        # do the copy atom associated with index i.
        # cA:    (BLK_M, BLK_K)      => (blk_m, blk_k)
        # cB:    (BLK_N, BLK_K)      => (blk_n, blk_k)
        # tAcA:  (CPY, CPY_M, CPY_K) => (blk_m, blk_k)
        # tBcB:  (CPY, CPY_N, CPY_K) => (blk_n, blk_k)
        # tApA: (rest_v, CPY_M, CPY_K), stride=(..., ..., 0)
        # tBpB: (rest_v, CPY_N, CPY_K), stride=(..., ..., 0)
        # CPY =  (atom_v, rest_v)
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for sA and sB, used for predication
        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        cA = cute.local_tile(
            mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        cB = cute.local_tile(
            mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)
        # Repeat the partitioning with identity layouts
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)
        # Allocate predicate tensors for m and n
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        # Allocate predicate tensors for m, n and k for residue k-tile
        tApA_residue_k = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        # Set predicates for m/n bounds for mainloop
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        # Set predicates for m/n/k bounds for residue k tile
        for rest_v in range(tApA_residue_k.shape[0]):
            for m in range(tApA_residue_k.shape[1]):
                for k in range(tApA_residue_k.shape[2]):
                    coord_A = tAcA[(0, rest_v), m, k, 0]
                    tApA_residue_k[rest_v, m, k] = cute.elem_less(
                        (coord_A[0], cutlass.Int32(-1)), (mA.shape[0], coord_A[1])
                    )
        for rest_v in range(tBpB_residue_k.shape[0]):
            for n in range(tBpB_residue_k.shape[1]):
                for k in range(tBpB_residue_k.shape[2]):
                    coord_B = tBcB[(0, rest_v), n, k, 0]
                    tBpB_residue_k[rest_v, n, k] = cute.elem_less(
                        (coord_B[0], cutlass.Int32(-1)), (mB.shape[0], coord_B[1])
                    )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads for 0th k-tile, where we take care of the k-residue
        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue_k,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue_k,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )
        # Start async loads for 1st k-tile onwards, no k-residue handling needed
        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )

            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()

        # all tiles have been copied from global memory, so clear the
        # predicate tensor
        if k_tile_count < k_pipe_max:
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Define A/B partitioning and C accumulators.
        # ///////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        # Clear the accumulator
        tCrC.fill(0.0)

        # Current pipe index in smem to read from / write to
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)

        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]

        # ///////////////////////////////////////////////////////////////////////////////
        # PREFETCH register pipeline
        # ///////////////////////////////////////////////////////////////////////////////
        k_block_max = cute.size(tCrA, mode=[2])

        if k_block_max > 1:
            # Wait until our first prefetched tile is loaded in
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            self.cta_sync_barrier.arrive_and_wait()
            # Prefetch the first rmem from the first k-tile
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # 1. Shared memory pipeline (gmem -> smem):
        #    The default smem pipeline depth is 3, meaning that for shared
        # memory buffers, we allocate three times the size described by the
        # CTA tiler. We prefetch 2 of these buffers before entering the main
        # loop. Considering only the transfer from global memory to shared
        # memory, the general structure of the mainloop is:
        #   (1) copy k-tile from gmem to smem;
        #   (2) perform gemm computation on k-tile;
        #   (3) wait for the next copy to finish.
        #    The `cute.arch.cp_async_wait_group(num_smem_stages - 2)` command
        # waits for the number of unfinished 'copy' to be <= 1. The advantage
        # of this approach is that it allows for simultaneous production
        # (i.e., step (1)) and consumption (i.e., step (2)) of smem.
        #    A common misconception is to prefetch N buffers and rewrite
        # the pipeline logic to wait on N-1 pending copies. The disadvantage
        # of this approach is that it requires fully consuming a buffer in
        # order to open an empty buffer for the next copy.
        # 2. Register pipeline (smem -> register):
        #    Similarly, the register pipeline produces i+1, consumes i, and
        # produces i+2... Notably, i and i+1 do not use the same register,
        # eliminating dependencies on the same register for better parallelism.
        # 3. Combining the smem and register pipelines results in the mainloop.
        # ///////////////////////////////////////////////////////////////////////////////

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    self.cta_sync_barrier.arrive_and_wait()

                # Load A, B from shared memory to registers for k_block + 1
                k_block_next = (k_block + 1) % k_block_max  # static
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_p[None, None, k_block_next],
                    tCrB[None, None, k_block_next],
                )

                # Fetch next A: To better interleave global memory access and
                # compute instructions, we intentionally use the sequence:
                # copy A, perform GEMM, then copy B.
                if k_block == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        # Use predicates because the m-mode may be irregular
                        pred=tApA,
                    )

                # Thread-level register gemm for k_block
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

                # Fetch next B and update smem pipeline read/write
                if k_block == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        # Use predicates because the n-mode may be irregular
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    # After copying all tiles, we avoid clearing the predicate
                    # tensor in the `mainloop` to prevent increasing its
                    # instruction count. Instead, we continue copying the
                    # first tile, though it won't be used. The 0-th tile is not
                    # copied due to its irregular shape, which could lead to
                    # illegal memory accesses.
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(1)
                    )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # Applies the epilogue operation to the accumulated results and copies
        # them without vectorization.
        # ///////////////////////////////////////////////////////////////////////////////
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        tCrC.store(epilogue_op(tCrC.load()))

        # predicate
        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)
        return


def cutlass_gemm(state: bench.State) -> None:
    n = state.get_int64("N")
    r = state.get_int64("R")

    dt = np.float32
    A_h = np.random.randn(n, r).astype(dt)
    B_h = np.copy(A_h.mT)
    C_h = np.eye(n, dtype=dt)

    if n >= 1024:
        # allow more time for large inputs
        state.set_timeout(360)

    dev_id = state.get_device()
    cs = state.get_stream()
    s = as_bindings_Stream(cs)
    core_s = as_core_Stream(cs)

    A_d = core.DeviceMemoryResource(dev_id).allocate(A_h.nbytes, core_s)
    B_d = core.DeviceMemoryResource(dev_id).allocate(B_h.nbytes, core_s)
    C_d = core.DeviceMemoryResource(dev_id).allocate(C_h.nbytes, core_s)

    driver.cuMemcpyAsync(A_d.handle, A_h.ctypes.data, A_h.nbytes, s)
    driver.cuMemcpyAsync(B_d.handle, B_h.ctypes.data, B_h.nbytes, s)
    driver.cuMemcpyAsync(C_d.handle, C_h.ctypes.data, C_h.nbytes, s)

    A_cp = make_tensor(A_h, A_d, dev_id)
    B_cp = make_tensor(B_h, B_d, dev_id)
    C_cp = make_tensor(C_h, C_d, dev_id)

    sgemm = SGemm()
    _ = sgemm(A_cp, B_cp, C_cp, stream=s)

    def launcher(launch: bench.Launch) -> None:
        s = as_bindings_Stream(launch.get_stream())
        sgemm(A_cp, B_cp, C_cp, stream=s)

    state.exec(launcher)


if __name__ == "__main__":
    gemm_b = bench.register(cutlass_gemm)
    gemm_b.add_int64_axis("R", [16, 64, 256])
    gemm_b.add_int64_axis("N", [256, 512, 1024, 2048])

    bench.run_all_benchmarks(sys.argv)
