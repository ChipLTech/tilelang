"""DLC-specific tile operations and intrinsics.

This module provides DLC hardware-specific operations for TileLang,
similar to the Ascend tile operations. These operations map to DLC
hardware instructions for efficient computation.
"""

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion
from typing import Union
from tvm import tir
import math


# DLC Address Space Constants (from dlc-defs.h)
# These identify different memory regions in DLC architecture
SMEM = 0       # Shared Memory
HBM = 1        # High Bandwidth Memory (global memory)
VMEM = 2       # Vector Memory (local/scratch memory)
CMEM = 3       # Constant Memory
IMEM = 4       # Instruction Memory
SEMAPHORE = 5  # Semaphore Space (for sync flags)

# Null semaphore constant
NULL_SEMAPHORE = 0


def _get_buffer_info(br, mask: str):
    """Get buffer pointer and size information.
    
    Args:
        br: The input Buffer, BufferRegion, or BufferLoad.
        mask: Access mode (e.g., "r" for read, "w" for write).
    
    Returns:
        ptr: The underlying access pointer with the correct offset applied.
        size: The total number of elements in the data block.
    """
    from tvm.tir import BufferLoad
    
    if isinstance(br, BufferLoad):
        # BufferLoad from alloc_local - extract the buffer
        real_buffer = br.buffer
        ptr = real_buffer.access_ptr(mask)
        size = math.prod(real_buffer.shape)
        return ptr, size
    elif isinstance(br, BufferRegion):
        real_buffer = br.buffer
        indices = [x.min for x in br.region]
        offset = real_buffer.offset_of(indices)[0]
        ptr = real_buffer.access_ptr(mask, offset=offset)
        
        size = 1
        for r in br.region:
            size *= r.extent
        
        return ptr, size
    elif isinstance(br, Buffer):
        ptr = br.access_ptr(mask)
        size = math.prod(br.shape)
        return ptr, size
    else:
        raise TypeError(f"Unsupported type: {type(br)}")


def _dtype(buf):
    """Map TVM dtype to DLC C type."""
    type_map = {
        "float16": "half",
        "float32": "float",
        "int32": "int",
        "uint32": "uint32_t",
        "bfloat16": "bfloat16_t",
        "uint16": "uint16_t",
        "uint8": "uint8_t",
        "int8": "int8_t",
        "int16": "int16_t",
        "int64": "int64_t",
        "uint64": "uint64_t",
    }
    if isinstance(buf, BufferRegion):
        buf = buf.buffer
    return type_map[buf.dtype]


def add(
    dst: Union[Buffer, BufferRegion],
    src0: Union[Buffer, BufferRegion],
    src1: Union[Buffer, BufferRegion, PrimExpr],
):
    """Element-wise addition operation for DLC.
    
    Performs dst = src0 + src1 using DLC vector instructions.
    
    Args:
        dst: Destination buffer or buffer region.
        src0: First source operand (buffer or buffer region).
        src1: Second source operand (buffer, buffer region, or scalar).
    
    Returns:
        A TVM intrinsic call that performs the addition.
    """
    dst_ptr, dst_size = _get_buffer_info(dst, "w")
    src0_ptr, src0_size = _get_buffer_info(src0, "r")
    
    if isinstance(src1, (PrimExpr, int, float)):
        # Scalar addition
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dlc_add_scalar"),
            f"DLCAddScalar<{_dtype(dst)}>",
            dst_ptr,
            src0_ptr,
            src1,
            dst_size,
        )
    else:
        # Vector addition
        src1_ptr, src1_size = _get_buffer_info(src1, "r")
        assert dst_size == src0_size == src1_size, "Buffer sizes must match"
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dlc_add"),
            f"DLCAdd<{_dtype(dst)}>",
            dst_ptr,
            src0_ptr,
            src1_ptr,
            dst_size,
        )


def mul(
    dst: Union[Buffer, BufferRegion],
    src0: Union[Buffer, BufferRegion],
    src1: Union[Buffer, BufferRegion, PrimExpr],
):
    """Element-wise multiplication operation for DLC.
    
    Performs dst = src0 * src1 using DLC vector instructions.
    
    Args:
        dst: Destination buffer or buffer region.
        src0: First source operand (buffer or buffer region).
        src1: Second source operand (buffer, buffer region, or scalar).
    
    Returns:
        A TVM intrinsic call that performs the multiplication.
    """
    dst_ptr, dst_size = _get_buffer_info(dst, "w")
    src0_ptr, src0_size = _get_buffer_info(src0, "r")
    
    if isinstance(src1, (PrimExpr, int, float)):
        # Scalar multiplication
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dlc_mul_scalar"),
            f"DLCMulScalar<{_dtype(dst)}>",
            dst_ptr,
            src0_ptr,
            src1,
            dst_size,
        )
    else:
        # Vector multiplication
        src1_ptr, src1_size = _get_buffer_info(src1, "r")
        assert dst_size == src0_size == src1_size, "Buffer sizes must match"
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.dlc_mul"),
            f"DLCMul<{_dtype(dst)}>",
            dst_ptr,
            src0_ptr,
            src1_ptr,
            dst_size,
        )


def fill(buffer: Union[Buffer, BufferRegion], value: PrimExpr):
    """Fill a buffer with a specified value.
    
    Args:
        buffer: Buffer or buffer region to be filled.
        value: The value to fill the buffer with.
    
    Returns:
        A TVM intrinsic call that performs the fill operation.
    """
    ptr, size = _get_buffer_info(buffer, "w")
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_fill"),
        f"DLCFill<{_dtype(buffer)}>",
        ptr,
        value,
        size,
    )


def abs(
    dst: Union[Buffer, BufferRegion],
    src: Union[Buffer, BufferRegion],
):
    """Element-wise absolute value operation for DLC.
    
    Performs dst = abs(src) using DLC vector instructions.
    
    Args:
        dst: Destination buffer or buffer region.
        src: Source operand (buffer or buffer region).
    
    Returns:
        A TVM intrinsic call that performs the absolute value operation.
    """
    dst_ptr, dst_size = _get_buffer_info(dst, "w")
    src_ptr, src_size = _get_buffer_info(src, "r")
    assert dst_size == src_size, "Buffer sizes must match"
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_abs"),
        f"DLCAbs<{_dtype(dst)}>",
        dst_ptr,
        src_ptr,
        dst_size,
    )


def dma(
    src: Union[Buffer, BufferRegion],
    src_space: PrimExpr,
    dst: Union[Buffer, BufferRegion],
    dst_space: PrimExpr,
    size: PrimExpr,
    src_flag: Union[Buffer, PrimExpr, None] = None,
    dst_flag: Union[Buffer, PrimExpr, None] = None,
    src_stride: PrimExpr = 128,
    dst_stride: PrimExpr = 128,
):
    """DLC DMA operation - asynchronous data transfer.
    
    Based on dlc_dma_new() from DLC reference implementation.
    Sync flags are allocated separately and passed as pointers.
    
    Args:
        src: Source buffer or buffer region.
        src_space: Source address space (HBM=1, VMEM=2, SMEM=0, CMEM=3).
        dst: Destination buffer or buffer region.
        dst_space: Destination address space (HBM=1, VMEM=2, SMEM=0, CMEM=3).
        size: Number of bytes to transfer.
        src_flag: Source sync flag buffer/pointer (or None for NULL_SEMAPHORE).
        dst_flag: Destination sync flag buffer/pointer (or None for NULL_SEMAPHORE).
        src_stride: Source stride in bytes (default: 128).
        dst_stride: Destination stride in bytes (default: 128).
    
    Returns:
        A TVM intrinsic call that performs DMA (void return).
    """
    src_ptr, _ = _get_buffer_info(src, "r")
    dst_ptr, _ = _get_buffer_info(dst, "w")
    
    # Handle sync flags - convert None to NULL_SEMAPHORE (0)
    src_flag_ptr = src_flag.access_ptr("rw") if isinstance(src_flag, Buffer) else (src_flag if src_flag is not None else NULL_SEMAPHORE)
    dst_flag_ptr = dst_flag.access_ptr("rw") if isinstance(dst_flag, Buffer) else (dst_flag if dst_flag is not None else NULL_SEMAPHORE)
    
    return tir.call_intrin(
        "handle",  # Void return
        tir.op.Op.get("tl.dlc_dma"),
        src_ptr,
        src_space,
        dst_ptr,
        dst_space,
        size,
        src_stride,
        dst_stride,
        src_flag_ptr,
        dst_flag_ptr,
    )


def sync(flag: Union[Buffer, PrimExpr]):
    """Wait for DMA operation to complete (dlc_sync_new).
    
    Args:
        flag: Sync flag buffer or pointer.
    
    Returns:
        A TVM intrinsic call that performs synchronization.
    """
    flag_ptr = flag.access_ptr("r") if isinstance(flag, Buffer) else flag
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_sync"),
        flag_ptr,
    )


def sync_done(flag: Union[Buffer, PrimExpr]):
    """Wait for DMA operation to be done (dlc_sync_done_new).
    
    Args:
        flag: Sync flag buffer or pointer.
    
    Returns:
        A TVM intrinsic call that waits for completion.
    """
    flag_ptr = flag.access_ptr("r") if isinstance(flag, Buffer) else flag
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_sync_done"),
        flag_ptr,
    )


def sync_gte(flag: Union[Buffer, PrimExpr], threshold: PrimExpr):
    """Wait for DMA operation to reach a threshold (dlc_sync_gte_new).
    
    Args:
        flag: Sync flag buffer or pointer.
        threshold: Threshold value to wait for.
    
    Returns:
        A TVM intrinsic call that performs conditional synchronization.
    """
    flag_ptr = flag.access_ptr("r") if isinstance(flag, Buffer) else flag
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_sync_gte"),
        flag_ptr,
        threshold,
    )


def sync_clear(flag: Union[Buffer, PrimExpr]):
    """Clear a sync flag (dlc_sync_clear_new).
    
    Args:
        flag: Sync flag buffer or pointer to clear.
    
    Returns:
        A TVM intrinsic call that clears the flag.
    """
    flag_ptr = flag.access_ptr("rw") if isinstance(flag, Buffer) else flag
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_sync_clear"),
        flag_ptr,
    )


def alloc_sync_flag(size: int = 1):
    """Allocate synchronization flags for DMA operations.
    
    Allocates an array of sync flags in SEMAPHORE_SPACE.
    In the generated C code, these will be declared with SEMAPHORE_SPACE attribute.
    
    Args:
        size: Number of sync flags to allocate (default: 1).
    
    Returns:
        A buffer allocated in local scope (will be marked as SEMAPHORE_SPACE in codegen).
    """
    # Use T.alloc_local with local scope
    # TVM's "shared.barrier" is CUDA-specific and requires thread context
    # For DLC, use "local" scope and handle SEMAPHORE_SPACE in codegen
    from tilelang.language import alloc_local
    return alloc_local((size,), "int32", scope="local")


def copy(
    dst: Union[Buffer, BufferRegion],
    src: Union[Buffer, BufferRegion],
):
    """Copy data from source to destination using DLC DMA.
    
    This is a simplified copy operation that will be lowered to
    proper DMA operations during compilation.
    
    Args:
        dst: Destination buffer or buffer region.
        src: Source buffer or buffer region.
    
    Returns:
        A TVM intrinsic call that performs the DMA copy.
    """
    dst_ptr, dst_size = _get_buffer_info(dst, "w")
    src_ptr, src_size = _get_buffer_info(src, "r")
    
    assert dst_size == src_size, "Source and destination sizes must match"
    
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.dlc_copy"),
        f"DLCCopy<{_dtype(dst)}>",
        dst_ptr,
        src_ptr,
        dst_size,
    )
