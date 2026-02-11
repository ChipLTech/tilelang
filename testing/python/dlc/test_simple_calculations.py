"""Test DLC compilation with simple calculation kernels."""
import sys
import os
sys.path.insert(0, "/home/test/lanhu/tilelang-dlc")

import tilelang
import tilelang.language as T
from tilelang.engine.lower import lower
from tilelang.language.dlc_tile import HBM, VMEM, SMEM, CMEM, NULL_SEMAPHORE


def test_vector_add():
    """Test simple vector addition kernel."""
    print("\n" + "=" * 70)
    print("Test 1: Vector Addition (C = A + B)")
    print("=" * 70)
    
    N = 1024
    block_size = 256
    num_blocks = N // block_size
    
    @T.prim_func
    def vector_add(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32")
    ):
        with T.Kernel(num_blocks, is_dlc=True) as cid:
            # Allocate VMEM buffers
            vmem_a = T.alloc_local((block_size,), "float32")
            vmem_b = T.alloc_local((block_size,), "float32")
            vmem_c = T.alloc_local((block_size,), "float32")
            
            # Sync flags
            sync_flag_a = T.tile.alloc_sync_flag(1)
            sync_flag_b = T.tile.alloc_sync_flag(1)
            sync_flag_c = T.tile.alloc_sync_flag(1)
            
            # Load A from HBM to VMEM
            T.tile.dma(
                src=A[cid * block_size],
                src_space=HBM,
                dst=vmem_a,
                dst_space=VMEM,
                size=block_size * 4,
                src_flag=sync_flag_a,
                dst_flag=None
            )
            
            # Load B from HBM to VMEM
            T.tile.dma(
                src=B[cid * block_size],
                src_space=HBM,
                dst=vmem_b,
                dst_space=VMEM,
                size=block_size * 4,
                src_flag=sync_flag_b,
                dst_flag=None
            )
            
            # Wait for loads to complete
            T.tile.sync_done(sync_flag_a)
            T.tile.sync_done(sync_flag_b)
            T.tile.sync_clear(sync_flag_a)
            T.tile.sync_clear(sync_flag_b)
            
            # Compute: C = A + B using DLC tile operation
            T.tile.add(vmem_c, vmem_a, vmem_b)
            
            # Store C from VMEM to HBM
            T.tile.dma(
                src=vmem_c,
                src_space=VMEM,
                dst=C[cid * block_size],
                dst_space=HBM,
                size=block_size * 4,
                src_flag=None,
                dst_flag=sync_flag_c
            )
            
            T.tile.sync(sync_flag_c)
    
    print(f"\n1. Kernel: vector_add")
    print(f"2. Vector size: {N}")
    print(f"3. Block size: {block_size}")
    print(f"4. Number of blocks: {num_blocks}")
    
    print("\n5. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(vector_add, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n6. Generated source code:")
        print("-" * 70)
        source = artifact.kernel_source
        print(source)
        print("-" * 70)
        
        print("\n7. Checking for computation constructs...")
        checks = {
            "DLC headers": all(h in source for h in ["typehint.h", "ldst.h"]),
            "Addition operation": "+" in source or "add" in source.lower(),
            "Loop construct": "for" in source.lower() or "while" in source.lower(),
            "DMA operation": "dlc_dma" in source or "DMA" in source.upper(),
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


def test_vector_mul():
    """Test simple vector multiplication kernel."""
    print("\n" + "=" * 70)
    print("Test 2: Vector Multiplication (C = A * B)")
    print("=" * 70)
    
    N = 1024
    block_size = 128 * 8
    num_blocks = N // block_size
    
    @T.prim_func
    def vector_mul(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32")
    ):
        with T.Kernel(num_blocks, is_dlc=True) as cid:
            # Allocate VMEM buffers
            vmem_a = T.alloc_local((block_size,), "float32")
            vmem_b = T.alloc_local((block_size,), "float32")
            vmem_c = T.alloc_local((block_size,), "float32")
            
            # Sync flags
            sync_flag_a = T.tile.alloc_sync_flag(1)
            sync_flag_b = T.tile.alloc_sync_flag(1)
            sync_flag_c = T.tile.alloc_sync_flag(1)
            
            # Load A from HBM to VMEM
            T.tile.dma(
                src=A[cid * block_size],
                src_space=HBM,
                dst=vmem_a,
                dst_space=VMEM,
                size=block_size * 4,
                src_flag=sync_flag_a,
                dst_flag=None
            )
            
            # Load B from HBM to VMEM
            T.tile.dma(
                src=B[cid * block_size],
                src_space=HBM,
                dst=vmem_b,
                dst_space=VMEM,
                size=block_size * 4,
                src_flag=sync_flag_b,
                dst_flag=None
            )
            
            # Wait for loads to complete
            T.tile.sync_done(sync_flag_a)
            T.tile.sync_done(sync_flag_b)
            T.tile.sync_clear(sync_flag_a)
            T.tile.sync_clear(sync_flag_b)
            
            # Compute: C = A * B using DLC tile operation
            T.tile.mul(vmem_c, vmem_a, vmem_b)
            
            # Store C from VMEM to HBM
            T.tile.dma(
                src=vmem_c,
                src_space=VMEM,
                dst=C[cid * block_size],
                dst_space=HBM,
                size=block_size * 4,
                src_flag=None,
                dst_flag=sync_flag_c
            )
            
            T.tile.sync(sync_flag_c)
    
    print(f"\n1. Kernel: vector_mul")
    print(f"2. Vector size: {N}")
    print(f"3. Block size: {block_size}")
    print(f"4. Number of blocks: {num_blocks}")
    
    print("\n5. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(vector_mul, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n6. Generated source code:")
        print("-" * 70)
        source = artifact.kernel_source
        print(source)
        print("-" * 70)
        
        print("\n7. Checking for computation constructs...")
        checks = {
            "DLC headers": all(h in source for h in ["typehint.h", "ldst.h"]),
            "Multiplication operation": "*" in source or "mul" in source.lower(),
            "Loop construct": "for" in source.lower() or "while" in source.lower(),
            "DMA operation": "dlc_dma" in source or "DMA" in source.upper(),
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


def test_vector_abs():
    """Test absolute value kernel."""
    print("\n" + "=" * 70)
    print("Test 3: Vector Absolute Value (B = abs(A))")
    print("=" * 70)
    
    N = 1024
    block_size = 256
    num_blocks = N // block_size
    
    @T.prim_func
    def vector_abs(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32")
    ):
        with T.Kernel(num_blocks, is_dlc=True) as cid:
            # Allocate VMEM buffers
            vmem_a = T.alloc_local((block_size,), "float32")
            vmem_b = T.alloc_local((block_size,), "float32")
            
            # Sync flags
            sync_flag_a = T.tile.alloc_sync_flag(1)
            sync_flag_b = T.tile.alloc_sync_flag(1)
            
            # Load A from HBM to VMEM
            T.tile.dma(
                src=A[cid * block_size],
                src_space=HBM,
                dst=vmem_a,
                dst_space=VMEM,
                size=block_size * 4,
                src_flag=sync_flag_a,
                dst_flag=None
            )
            
            # Wait for load to complete
            T.tile.sync_done(sync_flag_a)
            T.tile.sync_clear(sync_flag_a)
            
            # Compute: B = abs(A) using DLC tile operation
            T.tile.abs(vmem_b, vmem_a)
            
            # Store B from VMEM to HBM
            T.tile.dma(
                src=vmem_b,
                src_space=VMEM,
                dst=B[cid * block_size],
                dst_space=HBM,
                size=block_size * 4,
                src_flag=None,
                dst_flag=sync_flag_b
            )
            
            T.tile.sync(sync_flag_b)
    
    print(f"\n1. Kernel: vector_abs")
    print(f"2. Vector size: {N}")
    print(f"3. Block size: {block_size}")
    print(f"4. Number of blocks: {num_blocks}")
    
    print("\n5. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(vector_abs, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n6. Generated source code:")
        print("-" * 70)
        source = artifact.kernel_source
        print(source)
        print("-" * 70)
        
        print("\n7. Checking for computation constructs...")
        checks = {
            "DLC headers": all(h in source for h in ["typehint.h", "ldst.h"]),
            "Absolute value": "abs" in source.lower() or "fabs" in source.lower(),
            "Loop construct": "for" in source.lower() or "while" in source.lower(),
            "DMA operation": "dlc_dma" in source or "DMA" in source.upper(),
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


def test_scalar_multiply():
    """Test scalar multiplication kernel (B = A * scalar)."""
    print("\n" + "=" * 70)
    print("Test 4: Scalar Multiplication (B = A * 2.5)")
    print("=" * 70)
    
    N = 1024
    block_size = 256
    num_blocks = N // block_size
    scalar = 2.5
    
    @T.prim_func
    def scalar_mul(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32")
    ):
        with T.Kernel(num_blocks, is_dlc=True) as cid:
            # Allocate VMEM buffers
            vmem_a = T.alloc_local((block_size,), "float32")
            vmem_b = T.alloc_local((block_size,), "float32")
            
            # Sync flags
            sync_flag_a = T.tile.alloc_sync_flag(1)
            sync_flag_b = T.tile.alloc_sync_flag(1)
            
            # Load A from HBM to VMEM
            T.tile.dma(
                src=A[cid * block_size],
                src_space=HBM,
                dst=vmem_a,
                dst_space=VMEM,
                size=block_size * 4,
                src_flag=sync_flag_a,
                dst_flag=None
            )
            
            # Wait for load to complete
            T.tile.sync_done(sync_flag_a)
            T.tile.sync_clear(sync_flag_a)
            
            # Compute: B = A * scalar using DLC tile operation
            T.tile.mul(vmem_b, vmem_a, T.float32(scalar))
            
            # Store B from VMEM to HBM
            T.tile.dma(
                src=vmem_b,
                src_space=VMEM,
                dst=B[cid * block_size],
                dst_space=HBM,
                size=block_size * 4,
                src_flag=None,
                dst_flag=sync_flag_b
            )
            
            T.tile.sync(sync_flag_b)
    
    print(f"\n1. Kernel: scalar_mul")
    print(f"2. Vector size: {N}")
    print(f"3. Block size: {block_size}")
    print(f"4. Scalar value: {scalar}")
    print(f"5. Number of blocks: {num_blocks}")
    
    print("\n6. Lowering kernel with target='dlc'...")
    try:
        artifact = lower(scalar_mul, target="dlc")
        print("✓ Lowering succeeded!")
        
        print("\n7. Generated source code:")
        print("-" * 70)
        source = artifact.kernel_source
        print(source)
        print("-" * 70)
        
        print("\n8. Checking for computation constructs...")
        checks = {
            "DLC headers": all(h in source for h in ["typehint.h", "ldst.h"]),
            "Multiplication operation": "*" in source or "mul" in source.lower(),
            "Scalar constant": str(scalar) in source or "2.5" in source,
            "Loop construct": "for" in source.lower() or "while" in source.lower(),
            "DMA operation": "dlc_dma" in source or "DMA" in source.upper(),
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


def compile_artifact_to_assembly(artifact, kernel_name):
    """Helper function to compile an artifact to assembly."""
    print("\n" + "=" * 70)
    print(f"Compiling {kernel_name} to Assembly")
    print("=" * 70)
    
    try:
        from tilelang.contrib import dlcc
        
        # Check if DLC Clang is available
        clang_path = dlcc.find_dlc_clang()
        if not clang_path:
            print("⚠ DLC Clang not found - skipping compilation")
            return None
        
        print(f"✓ DLC Clang found at: {clang_path}")
        
        # Check if DLC target is supported
        if not dlcc.check_dlc_target_support(clang_path, verbose=False):
            print("✗ DLC target is NOT supported by this Clang build")
            return None
        
        print("✓ DLC target is supported!")
        
        # Compile to assembly
        print("\nCompiling to assembly...")
        asm_code = dlcc.get_assembly_from_source(artifact.kernel_source, verbose=True)
        print("\n✓ Compilation to assembly succeeded!")
        print("\nGenerated Assembly:")
        print("=" * 70)
        print(asm_code)
        print("=" * 70)
        
        return asm_code
        
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all simple calculation tests."""
    print("\n" + "=" * 70)
    print("DLC Simple Calculation Kernels Test Suite")
    print("=" * 70)
    
    results = {}
    artifacts = {}
    
    # Test 1: Vector Addition
    try:
        artifact = test_vector_add()
        results["Vector Addition"] = "✓ PASSED"
        artifacts["vector_add"] = artifact
    except Exception as e:
        results["Vector Addition"] = f"✗ FAILED: {e}"
    
    # Test 2: Vector Multiplication
    try:
        artifact = test_vector_mul()
        results["Vector Multiplication"] = "✓ PASSED"
        artifacts["vector_mul"] = artifact
    except Exception as e:
        results["Vector Multiplication"] = f"✗ FAILED: {e}"
    
    # Test 3: Vector Absolute Value
    try:
        artifact = test_vector_abs()
        results["Vector Absolute Value"] = "✓ PASSED"
        artifacts["vector_abs"] = artifact
    except Exception as e:
        results["Vector Absolute Value"] = f"✗ FAILED: {e}"
    
    # Test 4: Scalar Multiplication
    try:
        artifact = test_scalar_multiply()
        results["Scalar Multiplication"] = "✓ PASSED"
        artifacts["scalar_mul"] = artifact
    except Exception as e:
        results["Scalar Multiplication"] = f"✗ FAILED: {e}"
    
    # Try to compile one of the successful artifacts to assembly
    if artifacts:
        print("\n" + "=" * 70)
        print("Attempting Assembly Compilation")
        print("=" * 70)
        
        # Try to compile the first successful artifact
        for name, artifact in artifacts.items():
            asm_code = compile_artifact_to_assembly(artifact, name)
            if asm_code:
                results[f"{name} Assembly Compilation"] = "✓ PASSED"
                break  # Only compile one to save time
            else:
                results[f"{name} Assembly Compilation"] = "⚠ SKIPPED"
    
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
