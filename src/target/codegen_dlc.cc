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
#include "../op/dlc.h"
#include "../support/ffi_aliases.h"

namespace tvm {
namespace codegen {

// Helper function to convert address space integer to DLC enum name
static std::string GetDLCAddressSpaceName(int space) {
  // enum { SMEM = 0, HBM = 1, VMEM = 2, CMEM = 3, IMEM = 4, SEMAPHORE = 5};
  switch (space) {
    case 0: return "SMEM";
    case 1: return "HBM";
    case 2: return "VMEM";
    case 3: return "CMEM";
    case 4: return "IMEM";
    case 5: return "SEMAPHORE";
    default: return std::to_string(space);
  }
}

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

void CodeGenTileLangDLC::VisitStmt_(const tir::AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  
  // Determine the storage scope
  auto it = alloc_storage_scope_.find(op->buffer_var.get());
  std::string scope = (it != alloc_storage_scope_.end()) ? it->second : "local";
  
  // Determine the address space attribute based on scope
  std::string addr_space_attr = "";
  if (scope == "local" || scope == "vmem") {
    addr_space_attr = " VMEM_SPACE ";
  } else if (scope == "semaphore") {
    addr_space_attr = " SEMAPHORE_SPACE ";
  }
  
  // Check if this is a semaphore/sync flag based on variable name
  if (vid.find("sync") != std::string::npos || vid.find("flag") != std::string::npos) {
    addr_space_attr = " SEMAPHORE_SPACE ";
  }
  
  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
  
  // Emit the type with address space attribute
  PrintType(op->dtype, stream);
  stream << addr_space_attr;
  stream << vid << "[" << constant_size << "];\n";
  
  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
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
  
  // Handle integer types - use 'int' instead of 'int32_t' for DLC
  if (t.is_int() && t.bits() == 32 && lanes == 1) {
    os << "int";
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

void CodeGenTileLangDLC::VisitExpr_(const tir::CallNode* op, std::ostream& os) {
  // Handle DLC-specific operations
  if (op->op.same_as(tl::dlc_dma())) {
    // DLC DMA operation: dlc_dma(src_ptr, src_space, dst_ptr, dst_space, size, src_stride, dst_stride, src_flag, dst_flag)
    // Maps to: dlc_dma_new(src_ptr, src_space, dst_ptr, dst_space, length, src_stride, dst_stride, sync_flag0, sync_flag1, unit_len, addr_unit_shift)
    ICHECK_EQ(op->args.size(), 9U);
    os << "dlc_dma_new(";
    PrintExpr(op->args[0], os);  // src_ptr
    os << ", ";
    // Convert src_space integer to enum name
    if (auto* src_space_int = op->args[1].as<tir::IntImmNode>()) {
      os << GetDLCAddressSpaceName(src_space_int->value);
    } else {
      PrintExpr(op->args[1], os);
    }
    os << ", ";
    PrintExpr(op->args[2], os);  // dst_ptr
    os << ", ";
    // Convert dst_space integer to enum name
    if (auto* dst_space_int = op->args[3].as<tir::IntImmNode>()) {
      os << GetDLCAddressSpaceName(dst_space_int->value);
    } else {
      PrintExpr(op->args[3], os);
    }
    os << ", ";
    PrintExpr(op->args[4], os);  // size/length
    os << ", ";
    PrintExpr(op->args[5], os);  // src_stride
    os << ", ";
    PrintExpr(op->args[6], os);  // dst_stride
    os << ", ";
    // sync_flag0 (src_flag) - emit NULL_SEMAPHORE if zero
    if (auto* flag0_int = op->args[7].as<tir::IntImmNode>()) {
      if (flag0_int->value == 0) {
        os << "NULL_SEMAPHORE";
      } else {
        PrintExpr(op->args[7], os);
      }
    } else {
      PrintExpr(op->args[7], os);
    }
    os << ", ";
    // sync_flag1 (dst_flag) - emit NULL_SEMAPHORE if zero
    if (auto* flag1_int = op->args[8].as<tir::IntImmNode>()) {
      if (flag1_int->value == 0) {
        os << "NULL_SEMAPHORE";
      } else {
        PrintExpr(op->args[8], os);
      }
    } else {
      PrintExpr(op->args[8], os);
    }
    os << ", 128, 2)";  // unit_len=128, addr_unit_shift=2 (for 4-byte float elements)
    return;
  } else if (op->op.same_as(tl::dlc_sync())) {
    // DLC sync operation: dlc_sync(tag)
    ICHECK_EQ(op->args.size(), 1U);
    os << "dlc_sync_new(";
    PrintExpr(op->args[0], os);
    os << ")";
    return;
  } else if (op->op.same_as(tl::dlc_sync_done())) {
    // DLC sync done check: dlc_sync_done(tag)
    ICHECK_EQ(op->args.size(), 1U);
    os << "dlc_sync_done_new(";
    PrintExpr(op->args[0], os);
    os << ")";
    return;
  } else if (op->op.same_as(tl::dlc_sync_gte())) {
    // DLC sync greater-than-or-equal: dlc_sync_gte(tag, value)
    ICHECK_EQ(op->args.size(), 2U);
    os << "dlc_sync_gte_new(";
    PrintExpr(op->args[0], os);
    os << ", ";
    PrintExpr(op->args[1], os);
    os << ")";
    return;
  } else if (op->op.same_as(tl::dlc_sync_clear())) {
    // DLC sync clear: dlc_sync_clear(tag)
    ICHECK_EQ(op->args.size(), 1U);
    os << "dlc_sync_clear_new(";
    PrintExpr(op->args[0], os);
    os << ")";
    return;
  } else if (op->op.same_as(tl::dlc_barrier())) {
    // DLC barrier: dlc_barrier()
    os << "barrier()";
    return;
  } else if (op->op.same_as(tl::dlc_copy())) {
    // DLC copy operation: dlc_copy(dst, src, size)
    ICHECK_EQ(op->args.size(), 3U);
    os << "vmem_copy(";
    PrintExpr(op->args[0], os);
    os << ", ";
    PrintExpr(op->args[1], os);
    os << ", ";
    PrintExpr(op->args[2], os);
    os << ")";
    return;
  } else if (op->op.same_as(tl::dlc_fill())) {
    // DLC fill operation: dlc_fill(dst, value, size)
    ICHECK_EQ(op->args.size(), 3U);
    os << "vmem_fill(";
    PrintExpr(op->args[0], os);
    os << ", ";
    PrintExpr(op->args[1], os);
    os << ", ";
    PrintExpr(op->args[2], os);
    os << ")";
    return;
  } else if (op->op.same_as(tl::dlc_add())) {
    EmitVectorBinaryOp("v_f32_add_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_add_scalar())) {
    EmitVectorScalarOp("v_f32_add_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_mul())) {
    EmitVectorBinaryOp("v_f32_mul_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_mul_scalar())) {
    EmitVectorScalarOp("v_f32_mul_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_sub())) {
    EmitVectorBinaryOp("v_f32_sub_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_sub_scalar())) {
    EmitVectorScalarOp("v_f32_sub_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_div())) {
    EmitVectorBinaryOp("v_f32_div_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_div_scalar())) {
    EmitVectorScalarOp("v_f32_div_b", op, os);
    return;
  } else if (op->op.same_as(tl::dlc_abs())) {
    EmitVectorUnaryOp("v_f32_abs", op, os);
    return;
  }
  
  // For other operations, delegate to parent
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangDLC::EmitVectorBinaryOp(const std::string& op_name, const tir::CallNode* op, std::ostream& os) {
  // Generate loop with vector intrinsics for binary operations
  // Args: template_str, dst_ptr, src0_ptr, src1_ptr, size
  ICHECK_EQ(op->args.size(), 5U);
  std::string var_name = name_supply_->FreshName("_dlc_vec");
  os << "{\n";
  PrintIndent();
  os << "  float8_128 " << var_name << "_x, " << var_name << "_y, " << var_name << "_o;\n";
  PrintIndent();
  os << "  for (int " << var_name << "_i = 0; " << var_name << "_i < ";
  PrintExpr(op->args[4], os);  // size
  os << "; " << var_name << "_i += 1024) {\n";
  PrintIndent();
  os << "    int " << var_name << "_len = min(";
  PrintExpr(op->args[4], os);  // size
  os << " - " << var_name << "_i, 1024);\n";
  PrintIndent();
  os << "    int " << var_name << "_mask = pre_exp2(" << var_name << "_len/128);\n";
  PrintIndent();
  os << "    " << var_name << "_x = v_f32_ld_tnsr_st_msk(" << var_name << "_i/32, ";
  PrintExpr(op->args[2], os);  // src0_ptr
  os << ", 1, " << var_name << "_mask);\n";
  PrintIndent();
  os << "    " << var_name << "_y = v_f32_ld_tnsr_st_msk(" << var_name << "_i/32, ";
  PrintExpr(op->args[3], os);  // src1_ptr
  os << ", 1, " << var_name << "_mask);\n";
  PrintIndent();
  os << "    " << var_name << "_o = " << op_name << "(" << var_name << "_x, " << var_name << "_y);\n";
  PrintIndent();
  os << "    v_f32_st_tnsr_st_msk(" << var_name << "_i/32, ";
  PrintExpr(op->args[1], os);  // dst_ptr
  os << ", 1, " << var_name << "_mask, " << var_name << "_o);\n";
  PrintIndent();
  os << "  }\n";
  PrintIndent();
  os << "}";
}

void CodeGenTileLangDLC::EmitVectorScalarOp(const std::string& op_name, const tir::CallNode* op, std::ostream& os) {
  // Generate loop with vector intrinsics for scalar operations
  // Args: template_str, dst_ptr, src_ptr, scalar, size
  ICHECK_EQ(op->args.size(), 5U);
  std::string var_name = name_supply_->FreshName("_dlc_vec");
  os << "{\n";
  PrintIndent();
  os << "  float8_128 " << var_name << "_x, " << var_name << "_o;\n";
  PrintIndent();
  os << "  float8_128 " << var_name << "_scalar = ";
  PrintExpr(op->args[3], os);  // scalar
  os << ";\n";
  PrintIndent();
  os << "  for (int " << var_name << "_i = 0; " << var_name << "_i < ";
  PrintExpr(op->args[4], os);  // size
  os << "; " << var_name << "_i += 1024) {\n";
  PrintIndent();
  os << "    int " << var_name << "_len = min(";
  PrintExpr(op->args[4], os);  // size
  os << " - " << var_name << "_i, 1024);\n";
  PrintIndent();
  os << "    int " << var_name << "_mask = pre_exp2(" << var_name << "_len/128);\n";
  PrintIndent();
  os << "    " << var_name << "_x = v_f32_ld_tnsr_st_msk(" << var_name << "_i/32, ";
  PrintExpr(op->args[2], os);  // src_ptr
  os << ", 1, " << var_name << "_mask);\n";
  PrintIndent();
  os << "    " << var_name << "_o = " << op_name << "(" << var_name << "_x, " << var_name << "_scalar);\n";
  PrintIndent();
  os << "    v_f32_st_tnsr_st_msk(" << var_name << "_i/32, ";
  PrintExpr(op->args[1], os);  // dst_ptr
  os << ", 1, " << var_name << "_mask, " << var_name << "_o);\n";
  PrintIndent();
  os << "  }\n";
  PrintIndent();
  os << "}";
}

void CodeGenTileLangDLC::EmitVectorUnaryOp(const std::string& op_name, const tir::CallNode* op, std::ostream& os) {
  // Generate loop with vector intrinsics for unary operations
  // Args: template_str, dst_ptr, src_ptr, size (in elements)
  // Pattern: loop with step 32 (in 128-byte units), using v_f32_ld_tnsr_b/v_f32_st_tnsr_b
  ICHECK_EQ(op->args.size(), 4U);
  std::string var_name = name_supply_->FreshName("_dlc_vec");
  os << "{\n";
  PrintIndent();
  os << "  int " << var_name << "_size128b = ";
  PrintExpr(op->args[3], os);  // size in elements
  os << " / 32;\n";  // Convert to 128-byte units (32 float32 = 128 bytes)
  PrintIndent();
  os << "#pragma clang loop unroll_count(2)\n";
  PrintIndent();
  os << "  for (int " << var_name << "_vs = 0; " << var_name << "_vs < " << var_name << "_size128b; " << var_name << "_vs += 32) {\n";
  PrintIndent();
  os << "    float8_128 " << var_name << "_x = v_f32_ld_tnsr_b(" << var_name << "_vs, ";
  PrintExpr(op->args[2], os);  // src_ptr
  os << ");\n";
  PrintIndent();
  os << "    " << var_name << "_x = " << op_name << "(" << var_name << "_x);\n";
  PrintIndent();
  os << "    v_f32_st_tnsr_b(" << var_name << "_vs, ";
  PrintExpr(op->args[1], os);  // dst_ptr
  os << ", " << var_name << "_x);\n";
  PrintIndent();
  os << "  }\n";
  PrintIndent();
  os << "}";
}

std::string CodeGenTileLangDLC::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

}  // namespace codegen
}  // namespace tvm
