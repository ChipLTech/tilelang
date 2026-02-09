// Copyright (c) TileLang / DLC Support.
// Licensed under the MIT License.

/*!
 * \file tl/op/dlc.h
 * \brief Define DLC-related operators.
 *
 */

#ifndef TVM_TL_OP_DLC_H_
#define TVM_TL_OP_DLC_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

// DLC binary operations
TVM_DLL const Op &dlc_add();
TVM_DLL const Op &dlc_add_scalar();
TVM_DLL const Op &dlc_sub();
TVM_DLL const Op &dlc_sub_scalar();
TVM_DLL const Op &dlc_mul();
TVM_DLL const Op &dlc_mul_scalar();
TVM_DLL const Op &dlc_div();
TVM_DLL const Op &dlc_div_scalar();

// DLC unary operations
TVM_DLL const Op &dlc_abs();
TVM_DLL const Op &dlc_exp();
TVM_DLL const Op &dlc_log();
TVM_DLL const Op &dlc_sqrt();
TVM_DLL const Op &dlc_rsqrt();
TVM_DLL const Op &dlc_relu();

// DLC memory operations
TVM_DLL const Op &dlc_fill();
TVM_DLL const Op &dlc_copy();

// DLC DMA operations
TVM_DLL const Op &dlc_dma();

// DLC synchronization operations
TVM_DLL const Op &dlc_sync();
TVM_DLL const Op &dlc_sync_done();
TVM_DLL const Op &dlc_sync_gte();
TVM_DLL const Op &dlc_sync_clear();
TVM_DLL const Op &dlc_barrier();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_DLC_H_
