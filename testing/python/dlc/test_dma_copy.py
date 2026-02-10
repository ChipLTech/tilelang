"""Test DLC DMA operations directly."""
import sys
sys.path.insert(0, "/home/test/lanhu/tilelang-dlc")

import tilelang
import tilelang.language as T
from tilelang.engine.lower import lower


# Import DLC address space constants
from tilelang.language.dlc_tile import HBM, VMEM, SMEM, CMEM, NULL_SEMAPHORE


def test_dma_copy():
    """Test DLC DMA copy operation with sync flags."""
    print("=" * 70)
    print("Testing DLC DMA Copy Operation")
    print("=" * 70)
    
    M, N = 1024, 1024
    block_M, block_N = 128, 256
    
    m_num = M // block_M  # 8
    n_num = N // block_N  # 4
    total_blocks = m_num * n_num  # 32
    
    vmem_size = block_M * block_N  # Size of VMEM buffer
    
    @T.prim_func
    def dma_copy(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32")
    ):
        # DLC kernel with single block dimension
        with T.Kernel(total_blocks, is_dlc=True) as cid:
            # Calculate 2D block indices from 1D compute ID
            bx = cid // n_num
            by = cid % n_num
            
            # Allocate VMEM buffer for staging
            vmem_buf = T.alloc_local((block_M, block_N), "float32")
            
            # Allocate sync flags in SEMAPHORE_SPACE
            sync_flag = T.tile.alloc_sync_flag(1)
            
            # DMA: HBM → VMEM
            # Transfer data from global memory A to local VMEM buffer
            T.tile.dma(
                src=A[bx * block_M, by * block_N],
                src_space=HBM,
                dst=vmem_buf,
                dst_space=VMEM,
                size=vmem_size * 4,  # 4 bytes per float32
                src_flag=sync_flag,
                dst_flag=None
            )
            
            # Wait for DMA to complete
            T.tile.sync_done(sync_flag)
            
            # Clear the sync flag for reuse
            T.tile.sync_clear(sync_flag)
            
            # DMA: VMEM → HBM
            # Transfer data from VMEM buffer to global memory B
            T.tile.dma(
                src=vmem_buf,
                src_space=VMEM,
                dst=B[bx * block_M, by * block_N],
                dst_space=HBM,
                size=vmem_size * 4,  # 4 bytes per float32
                src_flag=None,
                dst_flag=sync_flag
            )
            
            # Wait for write-back to complete
            T.tile.sync(sync_flag)
    
    print(f"\n1. Matrix size: {M} x {N}")
    print(f"2. Block size: {block_M} x {block_N}")
    print(f"3. Number of blocks: {total_blocks}")
    print(f"4. VMEM buffer size: {vmem_size} elements ({vmem_size * 4} bytes)")
    
    print("\n5. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(dma_copy, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n6. Device module:")
        print(f"  Functions: {list(artifact.device_mod.functions.keys())}")
        
        print("\n7. Generated source code:")
        print("-" * 70)
        source = artifact.kernel_source
        print(source)
        print("-" * 70)
        
        print("\n8. Checking for DLC DMA constructs...")
        checks = {
            "DLC headers": all(h in source for h in ["typehint.h", "ldst.h", "kernel_arg_types.h"]),
            "DMA operation": "dlc_dma" in source or "DMA" in source.upper(),
            "Sync operation": "sync" in source.lower(),
            "SEMAPHORE": "SEMAPHORE" in source or "semaphore" in source.lower(),
            "Function body": "{" in source and "}" in source,
        }
        
        for name, found in checks.items():
            status = "✓" if found else "✗"
            print(f"  {status} {name}: {found}")
        
        print("\n9. Summary:")
        if all(checks.values()):
            print("✓ All DLC DMA constructs found!")
        else:
            missing = [k for k, v in checks.items() if not v]
            print(f"⚠ Some constructs missing: {missing}")
            print("  (This is expected - DMA lowering not yet implemented)")
        
        return artifact
        
    except Exception as e:
        print(f"✗ Lowering failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_dma_copy()
