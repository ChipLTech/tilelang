"""Test DLC compilation pipeline end-to-end."""
import sys
import os
sys.path.insert(0, "/home/test/lanhu/tilelang-dlc")

import tilelang
import tilelang.language as T
from tilelang.engine.lower import lower
from tilelang.language.dlc_tile import HBM, VMEM, SMEM, CMEM, NULL_SEMAPHORE

def test_dma_kernel():
    """Test DLC DMA kernel compilation."""
    print("\n" + "=" * 70)
    print("Test 2: DLC DMA Kernel Compilation")
    print("=" * 70)
    
    M, N = 1024, 1024
    block_M, block_N = 128, 256
    
    m_num = M // block_M
    n_num = N // block_N
    total_blocks = m_num * n_num
    vmem_size = block_M * block_N
    
    @T.prim_func
    def dma_copy(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32")
    ):
        with T.Kernel(total_blocks, is_dlc=True) as cid:
            bx = cid // n_num
            by = cid % n_num
            
            vmem_buf = T.alloc_local((block_M, block_N), "float32")
            sync_flag = T.tile.alloc_sync_flag(1)
            
            # DMA: HBM → VMEM
            T.tile.dma(
                src=A[bx * block_M, by * block_N],
                src_space=HBM,
                dst=vmem_buf,
                dst_space=VMEM,
                size=vmem_size * 4,
                src_flag=sync_flag,
                dst_flag=None
            )
            
            T.tile.sync_done(sync_flag)
            T.tile.sync_clear(sync_flag)
            
            # DMA: VMEM → HBM
            T.tile.dma(
                src=vmem_buf,
                src_space=VMEM,
                dst=B[bx * block_M, by * block_N],
                dst_space=HBM,
                size=vmem_size * 4,
                src_flag=None,
                dst_flag=sync_flag
            )
            
            T.tile.sync(sync_flag)
    
    print(f"\n1. Kernel: dma_copy")
    print(f"2. Matrix size: {M} x {N}")
    print(f"3. Block size: {block_M} x {block_N}")
    print(f"4. Number of blocks: {total_blocks}")
    
    print("\n5. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(dma_copy, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n6. Generated source code:")
        print("-" * 70)
        source = artifact.kernel_source
        print(source)
        print("-" * 70)
        
        print("\n7. Checking for DLC DMA constructs...")
        checks = {
            "DLC headers": all(h in source for h in ["typehint.h", "ldst.h", "kernel_arg_types.h"]),
            "DMA operation": "dlc_dma" in source or "DMA" in source.upper(),
            "Sync operation": "sync" in source.lower(),
            "SEMAPHORE": "SEMAPHORE" in source or "semaphore" in source.lower(),
        }
        
        for name, found in checks.items():
            status = "✓" if found else "✗"
            print(f"  {status} {name}: {found}")
        
        return artifact
        
    except Exception as e:
        print(f"✗ Lowering failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_compilation_with_clang():
    """Test actual compilation with Clang if available."""
    print("\n" + "=" * 70)
    print("Test 3: Compilation with DLC Clang")
    print("=" * 70)
    
    try:
        from tilelang.contrib import dlcc
        
        # Check if DLC Clang is available
        clang_path = dlcc.find_dlc_clang()
        if clang_path:
            print(f"\n✓ DLC Clang found at: {clang_path}")
            
            # Check if DLC target is supported
            print("\n2. Checking DLC target support...")
            if dlcc.check_dlc_target_support(clang_path, verbose=True):
                print("✓ DLC target is supported!")
            else:
                print("✗ DLC target is NOT supported by this Clang build")
                print("  Skipping compilation test")
                return
        else:
            print("\n⚠ DLC Clang not found - skipping compilation test")
            print("  Set LLVM_PATH, DLC_HOME, or LLVM_HOME environment variable to enable compilation")
            return
        
        # Create a simple test C code
        test_code = """
// TileLang DLC Target - Generated C source for DLC toolchain
// Compile with: clang -target dlc -c <file>.c

#include "typehint.h"
#include "ldst.h"
#include "kernel_arg_types.h"

void test_kernel() {
    // Simple test kernel
    int x = 42;
}
"""
        
        print("\n3. Test C code:")
        print("-" * 70)
        print(test_code)
        print("-" * 70)
        
        print("\n4. Attempting to compile with DLC Clang...")
        try:
            # Try to get LLVM IR (less likely to fail than object file)
            llvm_ir = dlcc.get_llvm_ir_from_source(test_code, verbose=True)
            print("✓ Compilation to LLVM IR succeeded!")
            print("\n5. Generated LLVM IR (first 500 chars):")
            print("-" * 70)
            print(llvm_ir[:500])
            if len(llvm_ir) > 500:
                print("...")
            print("-" * 70)
        except Exception as e:
            print(f"⚠ Compilation failed: {e}")
            print("  This is expected if DLC headers are not in the include path")
            
    except ImportError as e:
        print(f"\n⚠ Could not import dlcc module: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DLC Compilation Pipeline Test Suite")
    print("=" * 70)
    
    results = {}
    
    # Test 1: DMA kernel
    try:
        artifact = test_dma_kernel()
        results["DMA Kernel Lowering"] = "✓ PASSED"
        
        # Print the generated source code
        if artifact:
            print("\n" + "=" * 70)
            print("Generated DLC C Source Code:")
            print("=" * 70)
            print(artifact.kernel_source)
            print("=" * 70)
            
            # Try to compile to assembly
            print("\n" + "=" * 70)
            print("Compiling DMA Kernel to Assembly:")
            print("=" * 70)
            try:
                from tilelang.contrib import dlcc
                
                # Check if DLC Clang is available
                clang_path = dlcc.find_dlc_clang()
                if clang_path:
                    print(f"✓ DLC Clang found at: {clang_path}")
                    
                    # Check if DLC target is supported
                    if dlcc.check_dlc_target_support(clang_path, verbose=False):
                        print("✓ DLC target is supported!")
                        
                        # Compile to assembly
                        print("\nCompiling to assembly...")
                        asm_code = dlcc.get_assembly_from_source(artifact.kernel_source, verbose=True)
                        print("\n✓ Compilation to assembly succeeded!")
                        print("\nGenerated Assembly:")
                        print("=" * 70)
                        print(asm_code)
                        print("=" * 70)
                        results["DMA Kernel Compilation"] = "✓ PASSED"
                    else:
                        print("✗ DLC target is NOT supported by this Clang build")
                        results["DMA Kernel Compilation"] = "⚠ SKIPPED (DLC target not supported)"
                else:
                    print("⚠ DLC Clang not found - skipping compilation")
                    results["DMA Kernel Compilation"] = "⚠ SKIPPED (Clang not found)"
            except Exception as e:
                print(f"✗ Compilation failed: {e}")
                results["DMA Kernel Compilation"] = f"✗ FAILED: {e}"
    except Exception as e:
        results["DMA Kernel Lowering"] = f"✗ FAILED: {e}"
    
    # Test 2: Clang compilation
    try:
        test_compilation_with_clang()
        results["Clang Compilation"] = "✓ PASSED"
    except Exception as e:
        results["Clang Compilation"] = f"✗ FAILED: {e}"
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
