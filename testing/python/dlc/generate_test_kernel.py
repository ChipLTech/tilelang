#!/usr/bin/env python3
"""Generate a test kernel for DLC Clang compilation"""

import tilelang.language as T
from tilelang.engine.lower import lower
from tilelang.language.dlc_tile import HBM, VMEM

N = 256

@T.prim_func
def vector_add(A: T.Tensor((N,), "float32"), 
               B: T.Tensor((N,), "float32"), 
               C: T.Tensor((N,), "float32")):
    with T.Kernel(1, is_dlc=True) as cid:
        vmem_a = T.alloc_local((N,), "float32")
        vmem_b = T.alloc_local((N,), "float32")
        vmem_c = T.alloc_local((N,), "float32")
        
        sync_flag = T.tile.alloc_sync_flag(1)
        
        # Load inputs from HBM to VMEM
        T.tile.dma(src=A, src_space=HBM,
                  dst=vmem_a, dst_space=VMEM,
                  size=N * 4, src_flag=sync_flag, dst_flag=None)
        T.tile.dma(src=B, src_space=HBM,
                  dst=vmem_b, dst_space=VMEM,
                  size=N * 4, src_flag=sync_flag, dst_flag=None)
        
        T.tile.sync_done(sync_flag)
        T.tile.sync_clear(sync_flag)
        
        # Compute using template
        T.tile.add(vmem_c, vmem_a, vmem_b)
        
        # Store result from VMEM to HBM
        T.tile.dma(src=vmem_c, src_space=VMEM,
                  dst=C, dst_space=HBM,
                  size=N * 4, src_flag=None, dst_flag=sync_flag)
        T.tile.sync(sync_flag)

artifact = lower(vector_add, target="dlc")

# Save to file
output_file = "/tmp/test_template_kernel.c"
with open(output_file, "w") as f:
    f.write(artifact.kernel_source)

print(f"Kernel saved to {output_file}")
print("\nGenerated code:")
print("=" * 80)
print(artifact.kernel_source)
print("=" * 80)
