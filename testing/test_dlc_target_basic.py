"""Test DLC target recognition."""
import sys
sys.path.insert(0, "/home/test/lanhu/tilelang-dlc")

from tilelang.utils.target import (
    determine_target,
    target_is_dlc,
    check_dlc_availability,
    SUPPORTED_TARGETS,
)
from tvm.target import Target


def test_dlc_in_supported_targets():
    """Test that DLC is listed in supported targets."""
    print("Testing DLC in SUPPORTED_TARGETS...")
    assert "dlc" in SUPPORTED_TARGETS
    assert "DLC" in SUPPORTED_TARGETS["dlc"]
    print("✓ DLC is in SUPPORTED_TARGETS")


def test_determine_target_explicit_dlc():
    """Test that explicit 'dlc' target is recognized."""
    print("\nTesting explicit DLC target...")
    target_str = determine_target("dlc")
    print(f"  determine_target('dlc') = {target_str}")
    assert target_str == "llvm --keys=dlc"
    
    # Test case-insensitive
    target_str2 = determine_target("DLC")
    print(f"  determine_target('DLC') = {target_str2}")
    assert target_str2 == "llvm --keys=dlc"
    print("✓ Explicit DLC target works")


def test_determine_target_dlc_object():
    """Test that DLC target can be created as Target object."""
    print("\nTesting DLC Target object...")
    target_obj = determine_target("dlc", return_object=True)
    print(f"  Target object: {target_obj}")
    print(f"  Target kind: {target_obj.kind.name}")
    print(f"  Target keys: {target_obj.keys}")
    assert isinstance(target_obj, Target)
    assert target_is_dlc(target_obj)
    print("✓ DLC Target object works")


def test_target_is_dlc_with_llvm_keys():
    """Test target_is_dlc recognizes llvm with dlc keys."""
    print("\nTesting target_is_dlc with llvm --keys=dlc...")
    target = Target("llvm --keys=dlc")
    print(f"  Target: {target}")
    print(f"  Kind: {target.kind.name}")
    print(f"  Keys: {target.keys}")
    assert target_is_dlc(target)
    print("✓ target_is_dlc recognizes llvm --keys=dlc")


def test_target_is_dlc_negative():
    """Test that non-DLC targets are not recognized as DLC."""
    print("\nTesting negative cases...")
    cuda_target = Target("cuda")
    assert not target_is_dlc(cuda_target)
    print("  ✓ CUDA target is not DLC")
    
    llvm_target = Target("llvm")
    assert not target_is_dlc(llvm_target)
    print("  ✓ Plain LLVM target is not DLC")


def test_check_dlc_availability():
    """Test DLC availability check."""
    print("\nTesting DLC availability check...")
    result = check_dlc_availability()
    print(f"  check_dlc_availability() = {result}")
    assert isinstance(result, bool)
    print("✓ DLC availability check works")


if __name__ == "__main__":
    print("=" * 60)
    print("DLC Target Recognition Tests")
    print("=" * 60)
    
    try:
        test_dlc_in_supported_targets()
        test_determine_target_explicit_dlc()
        test_determine_target_dlc_object()
        test_target_is_dlc_with_llvm_keys()
        test_target_is_dlc_negative()
        test_check_dlc_availability()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
