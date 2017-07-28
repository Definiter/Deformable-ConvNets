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

#ifndef MXNET_OPERATOR_CONTRIB_NN_ACTIVE_IM2COL_CUH_
#define MXNET_OPERATOR_CONTRIB_NN_ACTIVE_IM2COL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "../../mxnet_op.h"
#include "../../../common/cuda_utils.h"

namespace mxnet {
namespace op {

/*!
 * \brief active_im2col gpu kernel.
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename DType>
__global__ void active_im2col_gpu_kernel(const int n, const DType* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    DType* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    DType* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const DType* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : static_cast<DType>(0);
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

/*!
 * \brief DO NOT call this directly. Use wrapper function active_im2col() instead;
 */
template <typename DType>
inline void active_im2col_gpu(mshadow::Stream<gpu>* s,
                       const DType* data_im, const int channels,
                       const int height, const int width,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       DType* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  using namespace mxnet_op;
  // NOLINT_NEXT_LINE(whitespace/operators)
  active_im2col_gpu_kernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                             0, mshadow::Stream<gpu>::GetStream(s)>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  MSHADOW_CUDA_POST_KERNEL_CHECK(active_im2col_gpu_kernel);
}

/*!
 * \brief DO NOT call this directly. Use wrapper function active_col2im() instead;
 */
template <typename DType>
__global__ void active_col2im_gpu_kernel(const int n, const DType* data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    DType* data_im, OpReqType req) {
  CUDA_KERNEL_LOOP(index, n) {
    DType val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO(caffe): use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    KERNEL_ASSIGN(data_im[index], req, val);
  }
}


/*!\brief active_im2col gpu version
 * \param s device stream
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param col_shape column buffer shape (#channels, output_im_height, output_im_width, ...)
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_col column buffer pointer
 */
template <typename DType>
inline void active_im2col(mshadow::Stream<gpu>* s,
                   const DType* data_im, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_col) {
  // num_axes should be smaller than block size
  index_t num_spatial_axes = kernel_shape.ndim();
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  index_t num_kernels = im_shape[1] * col_shape.ProdShape(1, col_shape.ndim());
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 2:
    active_im2col_gpu_kernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1],
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], 
        col_shape[1], col_shape[2], data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
}

/*!
 * \brief DO NOT call this directly. Use wrapper function active_col2im() instead;
 */
template <typename DType>
inline void active_col2im_gpu(mshadow::Stream<gpu>* s, const DType* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    DType* data_im, OpReqType req) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  using namespace mxnet_op;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  active_col2im_gpu_kernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                             0, mshadow::Stream<gpu>::GetStream(s)>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im, req);
  MSHADOW_CUDA_POST_KERNEL_CHECK(col2im_gpu_kernel);
}


/*!\brief
 * gpu function of active_col2im algorithm
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
inline void active_col2im(mshadow::Stream<gpu>* s,
                   const DType* data_col, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_im, OpReqType req) {
  index_t num_spatial_axes = kernel_shape.ndim();
  index_t im_size = im_shape.ProdShape(1, im_shape.ndim());
  // num_axes should be smaller than block size
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 2:
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    active_col2im_gpu_kernel<DType><<<cuda_get_num_blocks(im_size), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
        im_size, data_col, im_shape[1], im_shape[2], im_shape[3],
        kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
        dilation[0], dilation[1], col_shape[1], col_shape[2], data_im, req);
    MSHADOW_CUDA_POST_KERNEL_CHECK(active_col2im_gpu_kernel);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_NN_ACTIVE_IM2COL_CUH_
