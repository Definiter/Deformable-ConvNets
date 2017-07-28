/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_concat-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConcatParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if ((1 == param.dim) &&
    (param.num_args < (dnnResourceMultipleDst - dnnResourceMultipleSrc))) {
    switch (dtype) {
      case mshadow::kFloat32:
      return new MKLConcatOp<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConcatOp<cpu, double>(param);
    default:
      break;
    }
  }
  if (enableMKLWarnGenerated())
    LOG(INFO) << MKLConcatOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new ConcatOp<cpu, DType>(param);
  });
  return op;
}

Operator* ConcatProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ConcatParam);

MXNET_REGISTER_OP_PROPERTY(Concat, ConcatProp)
.describe(R"code(Concate a list of array along a given axis.

The dimension sizes of the input arrays on the given axis should be the same.

For example::

  x = [[1,1],[1,1]]
  y = [[2,2],[2,2]]
  z = [[3,3],[3,3],[3,3]]

  Concat(x,y,z,dim=0) = [[ 1.,  1.],
                         [ 1.,  1.],
                         [ 2.,  2.],
                         [ 2.,  2.],
                         [ 3.,  3.],
                         [ 3.,  3.],
                         [ 3.,  3.]]

  Concat(x,y,z,dim=1) = [[ 1.,  1.,  2.,  2.],
                         [ 1.,  1.,  2.,  2.]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol[]", "List of tensors to concatenate")
.add_arguments(ConcatParam::__FIELDS__())
.set_key_var_num_args("num_args");

NNVM_REGISTER_OP(Concat).add_alias("concat");

}  // namespace op
}  // namespace mxnet
