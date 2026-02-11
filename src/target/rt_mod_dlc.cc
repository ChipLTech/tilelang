// Copyright (c) TileLang / DLC Support.
// DLC runtime module: build C source module for DLC toolchain.

#include "codegen_dlc.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include "../support/ffi_aliases.h"
#include "target/source/codegen_source_base.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Build a DLC C source module from an IRModule with compilation.
 * \param mod The IRModule containing DLC device functions.
 * \param target The DLC target.
 * \return A compiled DLC module (object file).
 */
ffi::Module BuildTileLangDLC(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangDLC cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangDLC: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  
  // Try to compile the code using the DLC compiler callback
  if (const auto f = ffi::Function::GetGlobal("tilelang_callback_dlc_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
    std::string compiled_obj = (*f)(code, target, pass_ctx->config).cast<std::string>();
    
    // For now, return a C source module with the compiled object
    // In the future, this could return a proper binary module
    return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
  } else {
    // If no compile callback is registered, just return the source
    return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
  }
}

/*!
 * \brief Build a DLC C source module from an IRModule without compilation.
 * \param mod The IRModule containing DLC device functions.
 * \param target The DLC target.
 * \return A C source module that can be compiled by the DLC toolchain.
 */
ffi::Module BuildTileLangDLCWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangDLC cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangDLC: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  // Return a C source module that can be compiled by DLC toolchain
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

// Register the DLC build functions
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_dlc", BuildTileLangDLC)
      .def("target.build.tilelang_dlc_without_compile", BuildTileLangDLCWithoutCompile);
}

}  // namespace codegen
}  // namespace tvm
