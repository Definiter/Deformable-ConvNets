/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 */

#ifndef MXNET_OPERATOR_CONTRIB_NN_ACTIVE_IM2COL_H_
#define MXNET_OPERATOR_CONTRIB_NN_ACTIVE_IM2COL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <cstring>
#include <vector>
#include "../../mxnet_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief cpu function of active_im2col algorithm
 * \param data_im pointer of an image (C, H, W,...) in the image batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_col start pointer of the column buffer to be filled
 */
template <typename DType>
inline void active_im2col(mshadow::Stream<cpu>* s,
                   const DType* data_im, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_col) {
  if (2 == kernel_shape.ndim()) {
    LOG(FATAL) << "only implemented in GPU";
  } else {
    LOG(FATAL) << "not implemented";
  }
}


/*!\brief
 * cpu function of active_col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType>
inline void active_col2im(mshadow::Stream<cpu>* s,
                   const DType* data_col, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_im, OpReqType req) {
  index_t num_spatial_axes = kernel_shape.ndim();
  LOG(FATAL) << "only implemented in GPU";
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./active_im2col.cuh"
#endif
#endif  // MXNET_OPERATOR_CONTRIB_NN_ACTIVE_IM2COL_H_
