import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
from test_optimizer import *
import mxnet as mx
import numpy as np
from mxnet.test_utils import check_consistency, set_default_context
from numpy.testing import assert_allclose
import time

set_default_context(mx.gpu(0))
del test_support_vector_machine_l1_svm
del test_support_vector_machine_l2_svm


def check_countsketch(in_dim,out_dim,n):
    sym = mx.contrib.sym.count_sketch(name='countsketch',out_dim = out_dim)
    shape = [(n,in_dim), (1,in_dim),(1,in_dim)]     #shape of input x, hash h and hash s

    arr = [mx.nd.empty(shape[i]) for i in range(3)]
    arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
    x = np.random.uniform(-10, 10, shape[0])
    arr[0][:] = x                                 #input x
    h = np.random.randint(0, out_dim, shape[1])
    arr[1][:] = h                                 #hash h
    s = np.random.randint(0, 2, shape[2])*2-np.ones(shape[2])
    arr[2][:] = s                                 #hash s
    # forward
    exe_list = [sym.bind(mx.gpu(0), arr, arr_grad)]
    for exe in exe_list:
        exe.forward(is_train= True)
    out1 = [exe.outputs[0].asnumpy() for exe in exe_list]
    
    a = np.zeros((n,out_dim))
    temp = np.multiply(x, s)
    for num_sample in np.arange(0,n):
        for idx in np.arange(0,in_dim):
            a[num_sample][h[0][idx]] += temp[num_sample][idx]
    assert_almost_equal(a,out1[0],rtol=1e-3, atol=1e-12)
    
    # backward
    out_grad = mx.nd.empty((n,out_dim))
    out_grad[:] = np.random.normal(-3, 3, (n,out_dim))
    for exe in exe_list:
        exe.backward([out_grad])  
    
        a = np.zeros((n,in_dim))
        for j in np.arange(0,n):
            for i in np.arange(0,in_dim):
                a[j,i] = out_grad.asnumpy()[j, h[0,i]] * s[0,i]
    assert_almost_equal(a,arr_grad[0].asnumpy(),rtol=1e-3, atol=1e-12)
    
def test_countsketch():
    np.random.seed(0)
    nrepeat = 2
    minindim = 40
    maxindim = 100
    minoutdim = 5
    maxoutdim = 30
    maxn = 200
    for repeat in range(nrepeat):
        in_dim = np.random.randint(minindim, maxindim)
        out_dim = np.random.randint(minoutdim, maxoutdim)
        n = np.random.randint(1,maxn)
        check_countsketch(in_dim, out_dim, n)

def check_ifft(shape):
    shape_old = shape
    if len(shape) == 2:
        if shape[1]%2 != 0:
            lst = list(shape)
            lst[1] = lst[1]*2
            shape = tuple(lst)
            shape_old = shape
        shape = (shape[0],shape[1]*2)
    if len(shape) == 4:
        if shape[3]%2 != 0:
            lst = list(shape)
            lst[3] = lst[3]*2
            shape = tuple(lst)
            shape_old = shape
        shape = (shape[0],shape[1],shape[2],shape[3]*2)
    sym = mx.contrib.sym.ifft(name='ifft', compute_size = 128)
    init = [np.random.normal(size=shape, scale=1.0)]
    arr_grad = [mx.nd.empty(shape)]
    ctx_list = [{'ctx': mx.gpu(0),'ifft_data': shape, 'type_dict': {'ifft_data': np.float32}}]
    exe_list = [sym.simple_bind(args_grad=arr_grad,**ctx) for ctx in ctx_list]
    
    for exe in exe_list:
        for arr, iarr in zip(exe.arg_arrays, init):
            arr[:] = iarr.astype(arr.dtype)
    # forward
    for exe in exe_list:
        exe.forward(is_train= True)
        out1 = [exe.outputs[0].asnumpy() for exe in exe_list]
    
    if len(shape) == 2:
        init_complex = np.zeros(shape_old,dtype = np.complex64)
        for i in range(0,shape_old[1]):
            init_complex.real[:,i] = init[0][:,2*i]
            init_complex.imag[:,i] = init[0][:,2*i+1]
        a = np.fft.ifft(init_complex, n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, out1[0]/shape_old[1],rtol=1e-3, atol=1e-12)
    
    if len(shape) == 4:
        init_complex = np.zeros(shape_old,dtype = np.complex64)
        for i in range(0,shape_old[3]):
            init_complex.real[:,:,:,i] = init[0][:,:,:,2*i]
            init_complex.imag[:,:,:,i] = init[0][:,:,:,2*i+1]
        a = np.fft.ifft(init_complex, n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, out1[0]/shape_old[3],rtol=1e-3, atol=1e-12)
    # backward
    if len(shape) == 2:
        out_grad = mx.nd.empty(shape_old)
        out_grad[:] = np.random.normal(-3, 3, shape_old)
        for exe in exe_list:
            exe.backward([out_grad])
            temp = exe.grad_arrays[0].asnumpy()
            temp = np.zeros(shape_old)
            for i in range(shape_old[1]):
                temp[:,i] = exe.grad_arrays[0].asnumpy()[:,2*i]
                
        a = np.fft.fft(out_grad.asnumpy(), n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, temp, rtol=1e-3, atol=1e-12)
    if len(shape) == 4:
        out_grad = mx.nd.empty(shape_old)
        out_grad[:] = np.random.normal(-3, 3, shape_old)
        for exe in exe_list:
            exe.backward([out_grad])
            temp = exe.grad_arrays[0].asnumpy()
            temp = np.zeros(shape_old)
            for i in range(shape_old[3]):
                temp[:,:,:,i] = exe.grad_arrays[0].asnumpy()[:,:,:,2*i]
                
        a = np.fft.fft(out_grad.asnumpy(), n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, temp, rtol=1e-3, atol=1e-12)
             
    
def test_ifft():
    np.random.seed(0)
    nrepeat = 2
    maxdim = 10
    for repeat in range(nrepeat):
        for order in [2,4]:
            shape = tuple(np.random.randint(1, maxdim, size=order))
            check_ifft(shape)

def check_fft(shape):
    sym = mx.contrib.sym.fft(name='fft', compute_size = 128)
    if len(shape) == 2:
        if shape[1]%2 != 0:
            lst = list(shape)
            lst[1] = lst[1]*2
            shape = tuple(lst)
            shape_old = shape
    if len(shape) == 4:
        if shape[3]%2 != 0:
            lst = list(shape)
            lst[3] = lst[3]*2
            shape = tuple(lst)
            shape_old = shape
    init = [np.random.normal(size=shape, scale=1.0)]
    arr_grad = [mx.nd.empty(shape)]
    ctx_list = [{'ctx': mx.gpu(0),'fft_data': shape, 'type_dict': {'fft_data': np.float32}}]
    exe_list = [sym.simple_bind(args_grad=arr_grad,**ctx) for ctx in ctx_list]
    
    for exe in exe_list:
        for arr, iarr in zip(exe.arg_arrays, init):
            arr[:] = iarr.astype(arr.dtype)
    #forward
    for exe in exe_list:
        exe.forward(is_train=True)
    out1 = [exe.outputs[0].asnumpy() for exe in exe_list]
    out = np.fft.fft(init, n=None, axis=-1, norm=None)
    if len(shape) == 2:
        out = np.reshape(out,(out.shape[1],out.shape[2]))
        out2 = np.append(out.real, out.imag, axis = 1)
        a = np.zeros(out1[0].shape)
        p = 0
        for i in range(out2.shape[1]//2):
            a[:,p] = out2[:,i]
            a[:,p+1] = out2[:,i+out2.shape[1]//2]
            p = p+2
   
    if len(shape) == 4:
        out = np.reshape(out,(out.shape[1],out.shape[2],out.shape[3],out.shape[4]))
        out2 = np.append(out.real, out.imag, axis = 1)
        a = np.zeros(out1[0].shape)
        for i in range(out1[0].shape[0]):
            for j in range(out1[0].shape[1]):
                p = 0
                for k in range(out2.shape[3]):
                    a[i,j,:,p] = out2[i,j,:,k]
                    a[i,j,:,p+1] = out2[i,j+out1[0].shape[1],:,k]
                    p = p+2
    
    assert_almost_equal(a, out1[0],rtol=1e-3, atol=1e-6)
    
    # backward
    if len(shape) == 2:
        out_grad = mx.nd.empty((shape[0],2*shape[1]))
        out_grad[:] = np.random.normal(-3, 3, (shape[0],2*shape[1]))
        # out_grad_to_complex
        out_grad_complex = np.zeros(shape,dtype = np.complex64)
        for i in range(0,shape[1]):
            out_grad_complex.real[:,i] = out_grad.asnumpy()[:,2*i]
            out_grad_complex.imag[:,i] = out_grad.asnumpy()[:,2*i+1]
        for exe in exe_list:
            exe.backward([out_grad])  
        a = np.fft.ifft(out_grad_complex, n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, exe.grad_arrays[0].asnumpy()/shape[1],rtol=1e-3, atol=1e-8)
         
    if len(shape) == 4:
        out_grad = mx.nd.empty(out1[0].shape)
        out_grad[:] = np.random.normal(-3, 3, out1[0].shape)
        # out_grad_to_complex
        out_grad_complex = np.zeros(shape,dtype = np.complex64)
        for i in range(0,shape[3]):
            out_grad_complex.real[:,:,:,i] = out_grad.asnumpy()[:,:,:,2*i]
            out_grad_complex.imag[:,:,:,i] = out_grad.asnumpy()[:,:,:,2*i+1]
        for exe in exe_list:
            exe.backward([out_grad])  
        a = np.fft.ifft(out_grad_complex, n=None, axis=-1, norm=None)
        assert_almost_equal(a.real, exe.grad_arrays[0].asnumpy()/shape[3],rtol=1e-3, atol=1e-6)

def test_fft():
    np.random.seed(0)
    nrepeat = 2
    maxdim = 10
    for repeat in range(nrepeat):
        for order in [2,4]:
            shape = tuple(np.random.randint(1, maxdim, size=order))
            check_fft(shape)

def test_batchnorm_with_type():
    sym = mx.sym.BatchNorm(name='norm', fix_gamma=False)
    ctx_list = [{'ctx': mx.gpu(0), 'norm_data': (10, 2, 10, 10), 'type_dict': {'norm_data': np.float32}},
                {'ctx': mx.cpu(0), 'norm_data': (10, 2, 10, 10), 'type_dict': {'norm_data': np.float32}}]
    check_consistency(sym, ctx_list)

    sym = mx.sym.BatchNorm(name='norm', fix_gamma=True)
    check_consistency(sym, ctx_list)

def test_convolution_with_type():
    sym1 = mx.sym.Convolution(num_filter=3, kernel=(3,3), name='conv')

    data = mx.sym.Variable('conv_data')
    w = mx.sym.Variable('conv_weight')
    b = mx.sym.Variable('conv_bias')
    w = mx.sym.transpose(w, axes=(0,2,3,1))
    sym2 = mx.sym.transpose(data, axes=(0,2,3,1))
    sym2 = mx.sym.Convolution(sym2, w, b, layout='NHWC', num_filter=3, kernel=(3,3))
    sym2 = mx.sym.transpose(sym2, axes=(0,3,1,2), name='conv')

    sym = [sym1, sym1, sym1, sym1, sym1, sym2, sym2]
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},
                # NHWC
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'conv_weight': (3, 2, 3, 3),
                 'type_dict': {'conv_data': np.float32, 'conv_weight': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'conv_weight': (3, 2, 3, 3),
                 'type_dict': {'conv_data': np.float16, 'conv_weight': np.float16}}
                ]
    check_consistency(sym, ctx_list)


def test_convolution_options():
    # 1D convolution
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7), 'type_dict': {'conv_data': np.float32}}]
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,), pad=(1,), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,), stride=(2,), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,), dilate=(2,), name='conv')
    check_consistency(sym, ctx_list)

    # 2D convolution
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float16}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), stride=(2,2), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), stride=(2,2), cudnn_off=True, name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), dilate=(2,2), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), dilate=(2,2), cudnn_off=True, name='conv')
    check_consistency(sym, ctx_list)

    # 3D convolution
    ctx_list = [{'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float64}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), cudnn_off=True, name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), name='conv')
    check_consistency(sym, ctx_list)
    sym = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), stride=(2,2,2), cudnn_off=True, name='conv')
    check_consistency(sym, ctx_list)


def test_convolution_versions():
    # 2D convolution NCHW
    ctx_list = [{'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    conv_v1_cpu = mx.sym.Convolution_v1(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    conv_v1_gpu = mx.sym.Convolution_v1(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv')
    conv_cudnn = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    conv_cpu = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), name='conv')
    conv_gpu = mx.sym.Convolution(num_filter=3, kernel=(3,3), pad=(1,1), cudnn_off=True, name='conv')
    syms = [conv_v1_cpu, conv_v1_gpu, conv_cudnn, conv_cpu, conv_gpu]
    check_consistency(syms, ctx_list)

    # 3D convolution NCDHW
    ctx_list = [{'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.cpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}},
                {'ctx': mx.gpu(0), 'conv_data': (2, 2, 5, 7, 7), 'type_dict': {'conv_data': np.float32}}]
    conv_cudnn = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    conv_cpu = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), name='conv')
    conv_gpu = mx.sym.Convolution(num_filter=3, kernel=(2,3,3), pad=(1,1,1), cudnn_off=True, name='conv')
    syms = [conv_cudnn, conv_cpu, conv_gpu]
    check_consistency(syms, ctx_list)


def test_pooling_with_type():
    ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float32}},
                {'ctx': mx.gpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float16}},
                {'ctx': mx.cpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.cpu(0), 'pool_data': (2, 2, 10, 10), 'type_dict': {'pool_data': np.float32}}]
    sym = mx.sym.Pooling(kernel=(3,3), pool_type='max', pooling_convention='valid', name='pool')
    check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(kernel=(3,3), pool_type='max', pooling_convention='full', name='pool')
    check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(kernel=(300,300), pool_type='max', global_pool=True, name='pool')
    check_consistency(sym, ctx_list)


def test_deconvolution_with_type():
    sym = mx.sym.Deconvolution(num_filter=2, kernel=(3,3), name='deconv')
    ctx_list = [{'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}},
                {'ctx': mx.gpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float16}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float64}},
                {'ctx': mx.cpu(0), 'deconv_data': (2, 2, 10, 10), 'type_dict': {'deconv_data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")


def test_bilinear_sampler_with_type():
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym = mx.sym.BilinearSampler(data=data, grid=grid)
    ctx_list = [{'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float64}},
                {'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float32}},
                {'ctx': mx.gpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float16}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float64}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")


def test_grid_generator_with_type():
    data = mx.sym.Variable('data')
    sym = mx.sym.GridGenerator(data=data, transform_type='affine', target_shape=(20, 20))
    ctx_list = [{'ctx': mx.gpu(0), 'data': (3, 6), 'type_dict': {'data': np.float32}},
                {'ctx': mx.cpu(0), 'data': (3, 6), 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")
    sym = mx.sym.GridGenerator(data=data, transform_type='warp', target_shape=(20, 20))
    ctx_list = [{'ctx': mx.gpu(0), 'data': (3, 2, 20, 20), 'type_dict': {'data': np.float32}},
                {'ctx': mx.cpu(0), 'data': (3, 2, 20, 20), 'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")


# Checking max pooling consistency over the data sets of different float types is problematic
# as one max value in a float32 data set may not be the max value in a float16 data set.
# This function will not be called.
def test_pooling_with_type():
    np.random.seed(1234)
    ctx_list = [{'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float32}},
                {'ctx': mx.gpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float16}},
                {'ctx': mx.cpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float64}},
                {'ctx': mx.cpu(0), 'pool_data': (10, 2, 10, 10), 'type_dict': {'pool_data': np.float32}}]

    sym = mx.sym.Pooling(name='pool', kernel=(3,3), stride=(2,2), pool_type='max')
    check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(name='pool', kernel=(3,3), pad=(1,1), pool_type='avg')
    check_consistency(sym, ctx_list)

    # this is unstable
    # sym = mx.sym.Pooling(name='pool', kernel=(5,5), pad=(2,2), pool_type='max')
    # check_consistency(sym, ctx_list)

    sym = mx.sym.Pooling(name='pool', kernel=(3,3), pad=(1,1), pool_type='sum')
    check_consistency(sym, ctx_list)


def test_pooling_versions():
    def test_pooling_versions_helper(pool_op_list, data, kernel, pool_type, pad, stride,
                                     pooling_convention='valid', global_pool=False):
        ctx_list = []
        sym_list = []
        # PoolingV1 cpu
        if 'pool_v1_cpu' in pool_op_list:
            ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
            if not global_pool:
                sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                  pooling_convention=pooling_convention, name='pool'))
            else:
                sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pool_type=pool_type, global_pool=True, name='pool'))
        # PoolingV1 gpu
        if 'pool_v1_gpu' in pool_op_list:
            ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
            if not global_pool:
                sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                                  pooling_convention=pooling_convention, name='pool'))
            else:
                sym_list.append(mx.sym.Pooling_v1(kernel=kernel, pool_type=pool_type, global_pool=True, name='pool'))
        # Pooling cpu
        if 'pool_cpu' in pool_op_list:
            ctx_list.append({'ctx': mx.cpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
            if not global_pool:
                sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                               pooling_convention=pooling_convention, name='pool'))
            else:
                sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type, global_pool=True, name='pool'))
        # Pooling gpu
        if 'pool_gpu' in pool_op_list:
            ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
            if not global_pool:
                sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                               pooling_convention=pooling_convention, cudnn_off=True, name='pool'))
            else:
                sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type, global_pool=True, cudnn_off=True,
                                               name='pool'))
        # CuDNNPooling
        if 'pool_cudnn' in pool_op_list:
            ctx_list.append({'ctx': mx.gpu(0), 'pool_data': data, 'type_dict': {'pool_data': np.float32}})
            if not global_pool:
                sym_list.append(mx.sym.Pooling(kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                               pooling_convention=pooling_convention, cudnn_off=False, name='pool'))
            else:
                sym_list.append(mx.sym.Pooling(kernel=kernel, pool_type=pool_type, global_pool=True, cudnn_off=False,
                                               name='pool'))
        check_consistency(sym_list, ctx_list)

    def test_1d_pooling(pool_type):
        data = (2, 3, 20)
        kernel = (4,)
        pad = (0,)
        stride = (1,)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='valid', global_pool=False)

        pad = (2,)
        stride = (2,)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='valid', global_pool=False)

        pad = (0,)
        stride = (1,)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='full', global_pool=False)

        pad = (2,)
        stride = (2,)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='full', global_pool=False)

        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     global_pool=True)

    def test_2d_pooling(pool_type):
        data = (2, 3, 20, 20)
        kernel = (4, 5)
        pad = (0, 0)
        stride = (1, 1)
        test_pooling_versions_helper(pool_op_list=['pool_v1_cpu', 'pool_v1_gpu', 'pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='valid', global_pool=False)

        # pool_v1 has bugs when pad is not 0, do not test PoolingV1 here
        pad = (2, 3)
        stride = (2, 3)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='valid', global_pool=False)

        pad = (0, 0)
        stride = (1, 1)
        test_pooling_versions_helper(pool_op_list=['pool_v1_cpu', 'pool_v1_gpu', 'pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='full', global_pool=False)

        # pool_v1 has bugs when pad is not 0, do not test PoolingV1 here
        pad = (2, 3)
        stride = (2, 3)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='full', global_pool=False)

        test_pooling_versions_helper(pool_op_list=['pool_v1_cpu', 'pool_v1_gpu', 'pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     global_pool=True)

    def test_3d_pooling(pool_type):
        data = (2, 3, 20, 20, 20)
        kernel = (4, 5, 3)
        pad = (0, 0, 0)
        stride = (1, 1, 1)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='valid', global_pool=False)

        pad = (2, 3, 3)
        stride = (2, 3, 1)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='valid', global_pool=False)

        pad = (0, 0, 0)
        stride = (1, 1, 1)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='full', global_pool=False)

        pad = (2, 3, 3)
        stride = (2, 3, 1)
        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     pooling_convention='full', global_pool=False)

        test_pooling_versions_helper(pool_op_list=['pool_cpu', 'pool_gpu', 'pool_cudnn'],
                                     data=data, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type,
                                     global_pool=True)

    test_1d_pooling('max')
    test_1d_pooling('avg')
    test_1d_pooling('sum')

    test_2d_pooling('max')
    test_2d_pooling('avg')
    test_2d_pooling('sum')

    test_3d_pooling('max')
    test_3d_pooling('avg')
    test_3d_pooling('sum')


def test_upsampling_with_type():
    sym = mx.sym.UpSampling(scale=2, num_filter=2, name='up', sample_type='nearest', num_args=1)
    ctx_list = [{'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float64}},
                {'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float32}},
                {'ctx': mx.gpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float16}},
                {'ctx': mx.cpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float64}},
                {'ctx': mx.cpu(0), 'up_arg0': (2, 2, 2, 10), 'type_dict': {'up_arg0': np.float32}}]
    check_consistency(sym, ctx_list)


def test_upsampling_bilinear_with_type():
    sym = mx.sym.UpSampling(scale=2, num_filter=2, name='up', sample_type='bilinear', num_args=1)
    ctx_list = [{'ctx': mx.gpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float64}},
                {'ctx': mx.gpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float32}},
                {'ctx': mx.gpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float16}},
                {'ctx': mx.cpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float64}},
                {'ctx': mx.cpu(0), 'up_data': (2, 2, 2, 10), 'type_dict': {'up_data': np.float32}}]
    check_consistency(sym, ctx_list)


def test_concat_with_type():
    sym = mx.sym.Concat(name='concat', num_args=2)
    ctx_list = [{'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},
                {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}},
                {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float16, 'concat_arg1': np.float16}},
                {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},
                {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),
                 'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}}]
    check_consistency(sym, ctx_list)


def test_elementwisesum_with_type():
    sym = mx.sym.ElementWiseSum(name='ews', num_args=2)
    ctx_list = [{'ctx': mx.gpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float64, 'ews_arg1': np.float64}},
                {'ctx': mx.gpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float32, 'ews_arg1': np.float32}},
                {'ctx': mx.gpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float16, 'ews_arg1': np.float16}},
                {'ctx': mx.cpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float64, 'ews_arg1': np.float64}},
                {'ctx': mx.cpu(0), 'ews_arg1': (2, 10), 'ews_arg0': (2, 10),
                 'type_dict': {'ews_arg0': np.float32, 'ews_arg1': np.float32}}]
    check_consistency(sym, ctx_list)


def test_reshape_with_type():
    sym = mx.sym.Reshape(name='reshape', shape=(-1,1,1,0))
    ctx_list = [{'ctx': mx.gpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float64}},
                {'ctx': mx.gpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float32}},
                {'ctx': mx.gpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float16}},
                {'ctx': mx.cpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float64}},
                {'ctx': mx.cpu(0), 'reshape_data': (2, 2, 2, 10), 'type_dict': {'reshape_data': np.float32}}]
    check_consistency(sym, ctx_list)


def test_blockgrad_with_type():
    sym = mx.sym.BlockGrad(name='bg')
    ctx_list = [{'ctx': mx.gpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float64}},
                {'ctx': mx.gpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float32}},
                {'ctx': mx.gpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float16}},
                {'ctx': mx.cpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float64}},
                {'ctx': mx.cpu(0), 'bg_data': (2, 2, 2, 10), 'type_dict': {'bg_data': np.float32}}]
    check_consistency(sym, ctx_list)


def test_swapaxis_with_type():
    sym = mx.sym.SwapAxis(name='swap', dim1=1)
    ctx_list = [{'ctx': mx.gpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float64}},
                {'ctx': mx.gpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float32}},
                {'ctx': mx.gpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float16}},
                {'ctx': mx.cpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float64}},
                {'ctx': mx.cpu(0), 'swap_data': (2, 2, 2, 10), 'type_dict': {'swap_data': np.float32}}]
    check_consistency(sym, ctx_list)


def test_fullyconnected_with_type():
    sym = mx.sym.FullyConnected(num_hidden=3, name='inner')
    ctx_list = [{'ctx': mx.gpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float64}},
                {'ctx': mx.gpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float32}},
                {'ctx': mx.gpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float16}},
                {'ctx': mx.cpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float64}},
                {'ctx': mx.cpu(0), 'inner_data': (2, 10), 'type_dict': {'inner_data': np.float32}}]
    check_consistency(sym, ctx_list)


def test_activation_with_type():
    sym = mx.sym.Activation(name='act', act_type='sigmoid')
    ctx_list = [{'ctx': mx.gpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float64}},
                {'ctx': mx.gpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float32}},
                {'ctx': mx.gpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float16}},
                {'ctx': mx.cpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float64}},
                {'ctx': mx.cpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float32}},
                {'ctx': mx.cpu(0), 'act_data': (2, 2, 10, 10), 'type_dict': {'act_data': np.float16}}]
    check_consistency(sym, ctx_list)


def test_embedding_with_type():
    sym = mx.sym.Embedding(name='embedding', input_dim=10, output_dim=20)
    ctx_list = [{'ctx': mx.gpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float64}},
                {'ctx': mx.gpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float32}},
                {'ctx': mx.gpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float16}},
                {'ctx': mx.cpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float64}},
                {'ctx': mx.cpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float32}},
                {'ctx': mx.cpu(0), 'embedding_data': (2, 10), 'type_dict': {'embedding_data': np.float16}}]
    arg_params = {'embedding_data': np.random.randint(low=0, high=10, size=(2, 10))}
    check_consistency(sym, ctx_list, grad_req={'embedding_data': 'null','embedding_weight': 'write'},
                      arg_params=arg_params)


def test_svmoutput_with_type():
    sym = mx.sym.SVMOutput(name='svmoutput', use_linear=True)
    ctx_list = [{'ctx': mx.gpu(0), 'svmoutput_data': (20, 10), 'type_dict': {'svmoutput_data': np.float64}},
                {'ctx': mx.gpu(0), 'svmoutput_data': (20, 10), 'type_dict': {'svmoutput_data': np.float32}},
                {'ctx': mx.gpu(0), 'svmoutput_data': (20, 10), 'type_dict': {'svmoutput_data': np.float16}},
                {'ctx': mx.cpu(0), 'svmoutput_data': (20, 10), 'type_dict': {'svmoutput_data': np.float64}},
                {'ctx': mx.cpu(0), 'svmoutput_data': (20, 10), 'type_dict': {'svmoutput_data': np.float32}},
                {'ctx': mx.cpu(0), 'svmoutput_data': (20, 10), 'type_dict': {'svmoutput_data': np.float16}}]
    check_consistency(sym, ctx_list)


def test_take_with_type():
    sym = mx.sym.take(name='take')
    for data_ndim in range(2, 5):
        for idx_ndim in range(1, 4):
            data_shape = ()
            for _ in range(data_ndim):
                data_shape += (np.random.randint(low=3, high=6), )
            idx_shape = ()
            for _ in range(idx_ndim):
                idx_shape += (np.random.randint(low=3, high=5), )
            ctx_list = [{'ctx': mx.gpu(0), 'take_indices': idx_shape,
                         'take_a': data_shape,
                         'type_dict': {'take_indices': np.float64,
                                       'take_a': np.float64}},
                        {'ctx': mx.gpu(0), 'take_indices': idx_shape,
                         'take_a': data_shape,
                         'type_dict': {'take_indices': np.float32,
                                       'take_a': np.float32}},
                        {'ctx': mx.gpu(0), 'take_indices': idx_shape,
                         'take_a': data_shape,
                         'type_dict': {'take_indices': np.float16,
                                       'take_a': np.float16}},
                        {'ctx': mx.cpu(0), 'take_indices': idx_shape,
                         'take_a': data_shape,
                         'type_dict': {'take_indices': np.float64,
                                       'take_a': np.float64}},
                        {'ctx': mx.cpu(0), 'take_indices': idx_shape,
                         'take_a': data_shape,
                         'type_dict': {'take_indices': np.float32,
                                       'take_a': np.float32}},
                        {'ctx': mx.cpu(0), 'take_indices': idx_shape,
                         'take_a': data_shape,
                         'type_dict': {'take_indices': np.float16,
                                       'take_a': np.float16}}]
            arg_params = {'take_indices': np.random.randint(low=0,
                                                            high=data_shape[0],
                                                            size=idx_shape),
                          'take_a': np.random.normal(size=data_shape)}
            check_consistency(sym, ctx_list,
                              grad_req={'take_indices': 'null',
                                        'take_a': 'write'},
                              arg_params=arg_params)


def check_rnn_consistency(cell1, cell2):
    dshape = (32, 5, 200)
    data = mx.sym.Variable('data')

    sym1, _ = cell1.unroll(5, data, merge_outputs=True)
    mod1 = mx.mod.Module(sym1, label_names=None, context=mx.gpu(0))
    mod1.bind(data_shapes=[('data', dshape)], label_shapes=None)

    sym2, _ = cell2.unroll(5, data, merge_outputs=True)
    mod2 = mx.mod.Module(sym2, label_names=None, context=mx.gpu(0))
    mod2.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod1.init_params()
    args, auxs = mod1.get_params()
    args = cell1.unpack_weights(args)
    args = cell2.pack_weights(args)
    mod2.set_params(args, auxs)

    batch=mx.io.DataBatch(data=[mx.random.uniform(shape=dshape)], label=[])
    mod1.forward(batch, is_train=False)
    mod2.forward(batch, is_train=False)

    assert_allclose(mod1.get_outputs()[0].asnumpy(), mod2.get_outputs()[0].asnumpy(), rtol=1e-2, atol=1e-4)


def test_rnn():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='rnn_relu', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.RNNCell(100, activation='relu', prefix='l0_'))
    stack.add(mx.rnn.RNNCell(100, activation='relu', prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


def test_lstm():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='lstm', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.LSTMCell(100, prefix='l0_'))
    stack.add(mx.rnn.LSTMCell(100, prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


def test_lstm_forget_bias():
    forget_bias = 2.0
    fused = mx.rnn.FusedRNNCell(10, forget_bias=forget_bias, num_layers=2, mode='lstm', prefix='')

    dshape = (32, 1, 20)
    data = mx.sym.Variable('data')

    sym, _ = fused.unroll(1, data, merge_outputs=True)
    mod = mx.mod.Module(sym, label_names=None, context=mx.gpu(0))
    mod.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod.init_params()

    args, auxs = mod.get_params()
    args = fused.unpack_weights(args)

    bias_name = next(x for x in args if x.endswith('f_bias'))
    expected_bias = forget_bias * np.ones(10, )
    assert_allclose(args[bias_name].asnumpy(), expected_bias)


def test_gru():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='gru', prefix='')

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.GRUCell(100, prefix='l0_'))
    stack.add(mx.rnn.GRUCell(100, prefix='l1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)


def test_bidirectional():
    fused = mx.rnn.FusedRNNCell(100, num_layers=2, mode='gru', prefix='',
            bidirectional=True)

    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.BidirectionalCell(
                mx.rnn.GRUCell(100, prefix='l0_'),
                mx.rnn.GRUCell(100, prefix='r0_'),
                output_prefix='bi_gru_0_'))
    stack.add(mx.rnn.BidirectionalCell(
                mx.rnn.GRUCell(100, prefix='l1_'),
                mx.rnn.GRUCell(100, prefix='r1_'),
                output_prefix='bi_gru_1_'))

    check_rnn_consistency(fused, stack)
    check_rnn_consistency(stack, fused)

def test_unfuse():
    for mode in ['rnn_tanh', 'rnn_relu', 'lstm', 'gru']:
        fused = mx.rnn.FusedRNNCell(
            100, num_layers=2, mode=mode,
            prefix='test_%s'%mode,
            bidirectional=True,
            dropout=0.5)

        stack = fused.unfuse()

        check_rnn_consistency(fused, stack)
        check_rnn_consistency(stack, fused)

if __name__ == '__main__':
    test_countsketch()
    test_ifft()
    test_fft()
    test_bidirectional()
    test_lstm()
    test_lstm_forget_bias()
    test_gru()
    test_rnn()
    test_unfuse()
    test_convolution_options()
    test_convolution_versions()
    test_convolution_with_type()
    test_pooling_versions()
    test_batchnorm_with_type()
    test_batchnorm_with_type()
    test_deconvolution_with_type()
    test_upsampling_with_type()
    test_concat_with_type()
    test_elementwisesum_with_type()
    test_reshape_with_type()
    test_blockgrad_with_type()
    test_swapaxis_with_type()
    test_fullyconnected_with_type()
    test_activation_with_type()
    test_embedding_with_type()
    test_svmoutput_with_type()
    test_take_with_type()
    test_bilinear_sampler_with_type()
    test_grid_generator_with_type()
