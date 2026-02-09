// Copyright (c) TileLang / DLC Support.
// DLC runtime module: build C source module for DLC toolchain.

#include "codegen_dlc.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>

#include "../support/ffi_aliases.h"
#include "target/source/codegen_source_base.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Build a DLC C source module from an IRModule.
 * \param mod The IRModule containing DLC device functions.
 * \param target The DLC target.
 * \return A C source module that can be compiled by the DLC toolchain.
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
  // Return a C source module that can be compiled by DLC toolchain
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

// Register the DLC build function
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_dlc", BuildTileLangDLC);
}

}  // namespace codegen
}  // namespace tvm
