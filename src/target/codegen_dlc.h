// Copyright (c) TileLang / DLC Support.
// DLC codegen: emit C source for DLC toolchain.
#ifndef TILELANG_TARGET_CODEGEN_DLC_H_
#define TILELANG_TARGET_CODEGEN_DLC_H_

#include <string>
#include <vector>

#include "target/source/codegen_c.h"
#include "tvm/ir/attrs.h"
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Code generator for DLC accelerator.
 * Emits C source code that uses DLC builtins and memory hierarchy.
 */
class CodeGenTileLangDLC : public CodeGenC {
 public:
  CodeGenTileLangDLC();
  void Init(bool output_ssa);
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f);
  std::string Finish();
  ffi::Array<ffi::String> GetFunctionNames() const { return function_names_; }

  void PrintFuncPrefix(std::ostream& os) final;
  void VisitStmt_(const tir::AttrStmtNode* op) final;
  void VisitStmt_(const tir::AllocateNode* op) final;
  void VisitExpr_(const tir::CallNode* op, std::ostream& os) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;
  void PrintType(DataType t, std::ostream& os) final;

 private:
  void EmitVectorBinaryOp(const std::string& op_name, const tir::CallNode* op, std::ostream& os);
  void EmitVectorScalarOp(const std::string& op_name, const tir::CallNode* op, std::ostream& os);
  void EmitVectorUnaryOp(const std::string& op_name, const tir::CallNode* op, std::ostream& os);
  
  ffi::Array<ffi::String> function_names_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TILELANG_TARGET_CODEGEN_DLC_H_
