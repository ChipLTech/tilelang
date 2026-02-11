# pylint: disable=invalid-name
"""Utility to invoke DLC Clang compiler in the system"""

from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
import tvm_ffi
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.base import py_str
from tvm.contrib import utils


def find_dlc_clang() -> str | None:
    """Find the DLC Clang compiler.
    
    Returns
    -------
    str or None
        Path to the DLC clang compiler, or None if not found.
    """
    # Check LLVM_PATH environment variable first (as used in DLC_Custom_Kernel)
    llvm_path = os.environ.get("LLVM_PATH")
    if llvm_path:
        # Check build/bin/clang first, then bin/clang
        build_clang = os.path.join(llvm_path, "build", "bin", "clang")
        if os.path.exists(build_clang):
            return build_clang
        
        bin_clang = os.path.join(llvm_path, "bin", "clang")
        if os.path.exists(bin_clang):
            return bin_clang
    
    # Check DLC_HOME or LLVM_HOME
    dlc_home = os.environ.get("DLC_HOME") or os.environ.get("LLVM_HOME")
    if dlc_home:
        clang_path = os.path.join(dlc_home, "bin", "clang")
        if os.path.exists(clang_path):
            return clang_path
    
    # Check if clang is in PATH
    clang_path = shutil.which("clang")
    if clang_path:
        return clang_path
    
    # Check common installation paths
    common_paths = [
        "/usr/local/llvm/bin/clang",
        "/opt/llvm/bin/clang",
        "/usr/bin/clang",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def print_clang_targets(clang_path: str) -> str:
    """Print all targets supported by Clang.
    
    Parameters
    ----------
    clang_path : str
        Path to the Clang compiler.
        
    Returns
    -------
    str
        String containing all supported targets.
    """
    try:
        proc = subprocess.Popen(
            [clang_path, "-print-targets"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, _ = proc.communicate()
        return py_str(out)
    except Exception as e:
        return f"Error getting targets: {e}"


def check_dlc_target_support(clang_path: str, verbose: bool = False) -> bool:
    """Check if Clang supports the DLC target.
    
    Parameters
    ----------
    clang_path : str
        Path to the Clang compiler.
    verbose : bool
        If True, print all available targets.
        
    Returns
    -------
    bool
        True if DLC target is supported, False otherwise.
    """
    try:
        targets = print_clang_targets(clang_path)
        
        if verbose:
            print(f"Clang at {clang_path} supports the following targets:")
            print(targets)
        
        # Check if 'dlc' appears in the targets list
        # The output format is typically one target per line or space-separated
        return "dlc" in targets.lower()
    except Exception:
        return False


def get_dlc_clang_compiler() -> str:
    """Get the DLC Clang compiler path and verify DLC target support.
    
    Returns
    -------
    str
        Path to the DLC clang compiler.
        
    Raises
    ------
    RuntimeError
        If the DLC clang compiler is not found or doesn't support DLC target.
    """
    clang_path = find_dlc_clang()
    if clang_path is None:
        raise RuntimeError(
            "DLC Clang compiler not found. Please set LLVM_PATH, DLC_HOME, or LLVM_HOME "
            "environment variable, or ensure clang is in your PATH."
        )
    
    # Check if the compiler supports DLC target
    if not check_dlc_target_support(clang_path):
        raise RuntimeError(
            f"Clang compiler at {clang_path} does not support DLC target.\n"
            f"Please check your LLVM build and ensure it was built with DLC backend support.\n"
            f"You can verify by running: {clang_path} -print-targets | grep dlc"
        )
    
    return clang_path


def compile_dlc(
    code: str,
    target_format: str = "o",
    options: list[str] | str | None = None,
    path_target: str | None = None,
    verbose: bool = False,
) -> bytearray:
    """Compile DLC C code with Clang.
    
    Parameters
    ----------
    code : str
        The DLC C code to compile.
    target_format : str
        The target format: "o" (object file), "s" (assembly), "ll" (LLVM IR).
    options : str or list of str, optional
        Additional compiler options.
    path_target : str, optional
        Output file path. If None, a temporary file is used.
    verbose : bool
        Whether to print compiler output.
        
    Returns
    -------
    bytearray
        The compiled binary data.
        
    Raises
    ------
    RuntimeError
        If compilation fails.
    """
    if target_format not in ["o", "s", "ll", "bc"]:
        raise ValueError("target_format must be one of: o, s, ll, bc")
    
    temp = utils.tempdir()
    file_name = "dlc_kernel"
    temp_code = temp.relpath(f"{file_name}.c")
    temp_target = temp.relpath(f"{file_name}.{target_format}")
    
    # Check if there's a custom output directory from pass context
    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = pass_context.config.get("dlc.kernels_output_dir", None)
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.c")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")
    
    # Write source code to file
    with open(temp_code, "w") as out_file:
        out_file.write(code)
    
    file_target = path_target if path_target else temp_target
    
    # Build compiler command
    cmd = [get_dlc_clang_compiler()]
    
    # Add target triple for DLC
    cmd += ["-target", "dlc"]
    
    # Add optimization level
    cmd += ["-O2"]
    
    # Specify output format
    if target_format == "o":
        cmd += ["-c"]  # Compile only, don't link
    elif target_format == "s":
        cmd += ["-S"]  # Generate assembly
    elif target_format == "ll":
        cmd += ["-S", "-emit-llvm"]  # Generate LLVM IR
    elif target_format == "bc":
        cmd += ["-c", "-emit-llvm"]  # Generate LLVM bitcode
    
    # Add custom options
    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")
    
    # Add output file
    cmd += ["-o", file_target]
    
    # Add input file
    cmd += [temp_code]
    
    # Execute compiler
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    
    if verbose:
        print(py_str(out))
    
    if proc.returncode != 0:
        msg = f"{code}\nCompilation error:\n{py_str(out)}\nCommand: {' '.join(cmd)}\n"
        raise RuntimeError(msg)
    
    # Read compiled output
    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def default_compile_options(compile_flags: list[str] | None = None) -> list[str]:
    """Build a set of default DLC Clang compile options.
    
    Parameters
    ----------
    compile_flags : list of str, optional
        Additional flags to include.
        
    Returns
    -------
    list of str
        A list of flags suitable for DLC Clang's command line.
    """
    options: list[str] = []
    
    # Add C standard
    options.append("-std=dc99")
    
    # Add DLC kernel include paths (as used in DLC_Custom_Kernel)
    # 1. Synapse external includes
    synapse_include = "/usr/local/chipltech/synapse/include/external_includes"
    if os.path.exists(synapse_include):
        options.append(f"-I{synapse_include}")
    
    # 2. DLC Custom Kernel includes - try to find it dynamically
    # Check environment variable first
    dlc_custom_kernel_path = os.environ.get("DLC_CUSTOM_KERNEL_PATH")
    dlc_kernel_include_found = False
    
    if dlc_custom_kernel_path:
        dlc_kernel_include = os.path.join(dlc_custom_kernel_path, "dlc_kernels")
        if os.path.exists(dlc_kernel_include):
            options.append(f"-I{dlc_kernel_include}")
            dlc_kernel_include_found = True
        else:
            raise RuntimeError(
                f"DLC_CUSTOM_KERNEL_PATH is set to '{dlc_custom_kernel_path}' but "
                f"'{dlc_kernel_include}' does not exist."
            )
    else:
        # Try to find DLC_Custom_Kernel relative to current directory
        # Common locations to check
        search_paths = [
            os.path.join(os.getcwd(), "DLC_Custom_Kernel", "dlc_kernels"),
            os.path.join(os.path.dirname(os.getcwd()), "DLC_Custom_Kernel", "dlc_kernels"),
            os.path.join(os.path.expanduser("~"), "lanhu", "DLC_Custom_Kernel", "dlc_kernels"),
        ]
        for path in search_paths:
            if os.path.exists(path):
                options.append(f"-I{path}")
                dlc_kernel_include_found = True
                break
        
        if not dlc_kernel_include_found:
            raise RuntimeError(
                "DLC_Custom_Kernel/dlc_kernels directory not found. "
                "Please set the DLC_CUSTOM_KERNEL_PATH environment variable to the path "
                "containing DLC_Custom_Kernel directory.\n"
                f"Searched locations:\n" + "\n".join(f"  - {p}" for p in search_paths)
            )
    
    # Merge user-provided compile flags
    if compile_flags:
        import shlex
        for flag in compile_flags:
            tokens = shlex.split(flag) if isinstance(flag, str) else [str(flag)]
            options.extend(tokens)
    
    return options


def get_object_from_source(
    code: str,
    compile_flags: list[str] | None = None,
    verbose: bool = False,
) -> bytearray:
    """Compile DLC C source to object file.
    
    Parameters
    ----------
    code : str
        DLC C kernel source code.
    compile_flags : list of str, optional
        Additional flags merged with defaults.
    verbose : bool
        Print compiler output when True.
        
    Returns
    -------
    bytearray
        Object file binary data.
    """
    opts = default_compile_options(compile_flags)
    return compile_dlc(code, target_format="o", options=opts, verbose=verbose)


def get_assembly_from_source(
    code: str,
    compile_flags: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Compile DLC C source to assembly.
    
    Parameters
    ----------
    code : str
        DLC C kernel source code.
    compile_flags : list of str, optional
        Additional flags merged with defaults.
    verbose : bool
        Print compiler output when True.
        
    Returns
    -------
    str
        Assembly text.
    """
    opts = default_compile_options(compile_flags)
    asm_bytes = compile_dlc(code, target_format="s", options=opts, verbose=verbose)
    try:
        return asm_bytes.decode("utf-8")
    except Exception:
        return str(asm_bytes)


def get_llvm_ir_from_source(
    code: str,
    compile_flags: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Compile DLC C source to LLVM IR.
    
    Parameters
    ----------
    code : str
        DLC C kernel source code.
    compile_flags : list of str, optional
        Additional flags merged with defaults.
    verbose : bool
        Print compiler output when True.
        
    Returns
    -------
    str
        LLVM IR text.
    """
    opts = default_compile_options(compile_flags)
    ll_bytes = compile_dlc(code, target_format="ll", options=opts, verbose=verbose)
    try:
        return ll_bytes.decode("utf-8")
    except Exception:
        return str(ll_bytes)


@tvm_ffi.register_global_func("tilelang_callback_dlc_compile", override=True)
def tilelang_callback_dlc_compile(code, target, pass_config=None):
    """Callback function for DLC compilation.
    
    This function is called by TVM's build system to compile DLC C code.
    
    Parameters
    ----------
    code : str
        The DLC C source code.
    target : tvm.target.Target
        The compilation target.
    pass_config : dict, optional
        Pass configuration options.
        
    Returns
    -------
    bytearray
        The compiled object file.
    """
    cfg = pass_config or {}
    
    # Get compilation flags from config
    compile_flags = cfg.get("dlc.compile_flags", [])
    verbose = bool(cfg.get("dlc.verbose", False))
    
    # Build options
    options = default_compile_options(compile_flags)
    
    # Compile to object file
    return compile_dlc(code, target_format="o", options=options, verbose=verbose)
