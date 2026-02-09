"""
Element-wise addition kernel for DLC target.
This example demonstrates how to write a DLC kernel using TileLang.
"""
import argparse
import sys
sys.path.insert(0, "/home/test/lanhu/tilelang-dlc")

import tilelang
import tilelang.language as T

# Clear cache for fresh compilation
# tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="DLC Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n


@tilelang.jit(out_idx=[-1], target="dlc")
def vec_add_dlc(M, N, block_M, block_N, dtype="float32"):
    """Vector addition kernel for DLC.
    
    Args:
        M: Number of rows
        N: Number of columns
        block_M: Block size in M dimension
        block_N: Block size in N dimension
        dtype: Data type (default: float32)
    """
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # DLC kernel with single block dimension
        # DLC uses compute ID (cid) to identify the block
        with T.Kernel(m_num * n_num, is_dlc=True) as cid:
            # Calculate 2D block indices from 1D compute ID
            bx = cid // n_num
            by = cid % n_num
            
            # Allocate local buffers (VMEM in DLC)
            a_local = T.alloc_local((block_M, block_N), dtype)
            b_local = T.alloc_local((block_M, block_N), dtype)
            c_local = T.alloc_local((block_M, block_N), dtype)
            
            # Copy from global memory (HBM) to local memory (VMEM)
            T.copy(A[bx * block_M, by * block_N], a_local)
            T.copy(B[bx * block_M, by * block_N], b_local)
            
            # Perform element-wise addition using DLC tile operation
            T.tile.add(c_local, a_local, b_local)
            
            # Copy result back to global memory
            T.copy(c_local, C[bx * block_M, by * block_N])

    return main


# Compile the kernel
print("=" * 70)
print("Compiling DLC Vector Addition Kernel")
print("=" * 70)
print(f"Matrix size: {M} x {N}")
print(f"Block size: 128 x 256")

func = vec_add_dlc(M, N, 128, 256)

print("\n✓ Kernel compiled successfully!")
print("\nGenerated code can be found in the cache directory.")
print("To run on DLC hardware, integrate with DLC runtime.")
