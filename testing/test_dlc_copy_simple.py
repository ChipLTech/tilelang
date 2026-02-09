"""Test basic DLC copy (data transfer) operation."""
import sys
sys.path.insert(0, "/home/test/lanhu/tilelang-dlc")

import tilelang
import tilelang.language as T
from tilelang.engine.lower import lower


def test_simple_copy():
    """Test a simple copy operation from A to B."""
    print("=" * 60)
    print("Testing Simple Copy Operation for DLC")
    print("=" * 60)
    
    M, N = 1024, 1024
    block_M, block_N = 128, 256
    
    m_num = M // block_M  # 8
    n_num = N // block_N  # 4
    total_blocks = m_num * n_num  # 32
    
    @T.prim_func
    def simple_copy(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32")
    ):
        # DLC kernel with single block dimension
        with T.Kernel(total_blocks, is_dlc=True) as cid:
            # Calculate 2D block indices from 1D compute ID
            bx = cid // n_num
            by = cid % n_num
            
            # Simple direct copy - will be lowered to DMA later
            # For now, just do element-wise copy
            for i, j in T.grid(block_M, block_N):
                B[bx * block_M + i, by * block_N + j] = A[bx * block_M + i, by * block_N + j]
    
    print(f"\n1. Matrix size: {M} x {N}")
    print(f"2. Block size: {block_M} x {block_N}")
    print(f"3. Number of blocks: {total_blocks} (m_num={m_num}, n_num={n_num})")
    
    print("\n4. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(simple_copy, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n5. Device module:")
        print(f"  Functions: {list(artifact.device_mod.functions.keys())}")
        
        print("\n6. Generated source code:")
        print("-" * 60)
        source = artifact.kernel_source
        print(source)
        print("-" * 60)
        
        print("\n7. Checking for DLC-specific constructs...")
        checks = {
            "typehint.h": "typehint.h" in source,
            "ldst.h": "ldst.h" in source,
            "kernel_arg_types.h": "kernel_arg_types.h" in source,
            "DLC comment": "DLC" in source,
            "Function definition": "int32_t " in source or "void " in source or "int " in source,
            "Function body": "{" in source and "}" in source,
        }
        
        for name, found in checks.items():
            status = "✓" if found else "✗"
            print(f"  {status} {name}: {found}")
        
        print("\n8. Summary:")
        if all(checks.values()):
            print("✓ All DLC-specific constructs found!")
        else:
            missing = [k for k, v in checks.items() if not v]
            print(f"⚠ Missing constructs: {missing}")
        
        return artifact
        
    except Exception as e:
        print(f"✗ Lowering failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_simple_copy()
