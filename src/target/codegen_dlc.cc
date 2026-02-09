// Copyright (c) TileLang / DLC Support.
// DLC codegen: emit C source for DLC toolchain.

#include "codegen_dlc.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_set>

#include "../op/builtin.h"
#include "../support/ffi_aliases.h"

namespace tvm {
namespace codegen {

CodeGenTileLangDLC::CodeGenTileLangDLC() {}

void CodeGenTileLangDLC::Init(bool output_ssa) {
  function_names_.clear();
  decl_stream << "// TileLang DLC Target - Generated C source for DLC toolchain\n";
  decl_stream << "// Compile with: clang -target dlc -c <file>.c\n";
  decl_stream << "\n";
  decl_stream << "#include \"typehint.h\"\n";
  decl_stream << "#include \"ldst.h\"\n";
  decl_stream << "#include \"kernel_arg_types.h\"\n";
  decl_stream << "\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenTileLangDLC::PrintFuncPrefix(std::ostream& os) {
  // DLC kernels use plain C, no extern "C" needed
  // Function name is typically "main" for DLC kernels
}

void CodeGenTileLangDLC::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ffi::String func_name = global_symbol ? global_symbol.value() : gvar->name_hint;
  function_names_.push_back(func_name);

  this->InitFuncState(f);
  ReserveKeywordsAsUnique();

  std::unordered_set<const VarNode*> non_restrict;
  if (auto opt = f->GetAttr<ffi::Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tir::Var& v : opt.value()) {
      non_restrict.insert(v.get());
    }
  }
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  stream << " " << static_cast<std::string>(func_name) << "(";

  // Emit DLC-style parameters
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    
    if (v.dtype().is_handle()) {
      // For DLC, handle types become DLCTensor* or DLCScalar*
      // For now, emit as generic pointer - will enhance later
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          stream << "const ";
          CodeGenC::PrintType(ptr->element_type, stream);
          stream << "* " << vid;
          continue;
        }
      }
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }
      CodeGenC::PrintType(GetType(v), stream);
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
      if (no_alias && !non_restrict.count(v.get())) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << " " << vid;
  }
  stream << ") {\n";
  
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  stream << "}\n\n";
}

void CodeGenTileLangDLC::VisitStmt_(const tir::AttrStmtNode* op) {
  // Handle DLC-specific and common attributes
  if (op->attr_key == tir::attr::thread_extent) {
    // For DLC, we need to declare the thread variable as a regular variable
    // DLC uses a compute ID model, not CUDA-style grid/block threading
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      // Allocate variable ID for the thread variable
      std::string vid = AllocVarID(iv->var.get());
      
      // Emit variable declaration for DLC
      // For now, treat thread variables as regular int32 variables
      // In a real DLC kernel, these would be passed as parameters
      this->PrintIndent();
      this->PrintType(iv->var.dtype(), stream);
      stream << " " << vid << " = 0;  // Thread variable (extent: " << op->value << ")\n";
    }
    this->PrintStmt(op->body);
    return;
  } else if (op->attr_key == "storage_scope") {
    // Handle storage scope attributes (local, shared, etc.)
    this->PrintStmt(op->body);
    return;
  } else if (op->attr_key == "compute_scope") {
    // Handle compute scope attributes
    this->PrintStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") {
    // Handle resource scope attributes
    this->PrintStmt(op->body);
    return;
  }
  
  // For other attributes, delegate to parent
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangDLC::PrintStorageScope(const std::string& scope, std::ostream& os) {
  // DLC supports different memory scopes
  if (scope == "global") {
    // Global memory (HBM) - no prefix needed
  } else if (scope == "local") {
    // Local memory (VMEM) - no special prefix in C, just stack allocation
  } else if (scope == "shared" || scope == "shared.dyn") {
    // Shared memory - DLC may have specific handling
    // For now, treat as regular allocation
  } else {
    // Unknown scope - just continue without prefix
  }
}

void CodeGenTileLangDLC::PrintType(DataType t, std::ostream& os) {
  int lanes = t.lanes();
  
  if (t.is_handle()) {
    os << "void*";
    return;
  }
  
  if (t.is_void()) {
    os << "void";
    return;
  }
  
  // Handle vector types for DLC
  if (lanes > 1) {
    // DLC uses vector types like float4, int8_128, etc.
    // For now, emit as array type
    CodeGenC::PrintType(t.with_lanes(1), os);
    os << lanes;
    return;
  }
  
  // For scalar types, use parent implementation
  CodeGenC::PrintType(t, os);
}

std::string CodeGenTileLangDLC::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

}  // namespace codegen
}  // namespace tvm
