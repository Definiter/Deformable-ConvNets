# coding: utf-8
"""Autograd for NDArray."""
from __future__ import absolute_import
from __future__ import division

import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array
from ..ndarray import NDArray, zeros_like
from ..symbol import _GRAD_REQ_MAP

def set_recording(recording):
    """Turn on or turn off operator recording.

    Parameters
    ----------
    recording: bool
    """
    check_call(_LIB.MXAutogradSetRecording(
        ctypes.c_int(recording)))

def mark_variables(variables, gradients, grad_reqs='write'):
    """Mark NDArrays as variables to compute gradient for autograd.

    Parameters
    ----------
    variables: list of NDArray
    gradients: list of NDArray
    grad_reqs: list of string
    """
    variable_handles = []
    gradient_handles = []
    for var, gradvar in zip(variables, gradients):
        variable_handles.append(var.handle)
        gradient_handles.append(gradvar.handle)
    if isinstance(grad_reqs, string_types):
        grad_reqs = [_GRAD_REQ_MAP[grad_reqs]]*len(variables)
    else:
        grad_reqs = [_GRAD_REQ_MAP[i] for i in grad_reqs]

    check_call(_LIB.MXAutogradMarkVariables(
        len(variable_handles),
        c_array(NDArrayHandle, variable_handles),
        c_array(mx_uint, grad_reqs),
        c_array(NDArrayHandle, gradient_handles)))

def compute_gradient(outputs):
    """Compute the gradients of outputs w.r.t variables.

    Parameters
    ----------
    outputs: list of NDArray

    Returns
    -------
    gradients: list of NDArray
    """
    output_handles = []
    for arr in outputs:
        output_handles.append(arr.handle)

    check_call(_LIB.MXAutogradComputeGradient(
        len(output_handles),
        c_array(NDArrayHandle, output_handles)))


def grad_and_loss(func, argnum=None):
    """Return function that computes both gradient of arguments and loss value.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.
    argnum: an int or a list of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_and_loss_func: a python function
        A function that would compute both the gradient of arguments and loss value.
    """
    @functools.wraps(func)
    def wrapped(*args):
        """Wrapped function."""
        variables = args
        if argnum is not None:
            argnum_ = argnum if isinstance(argnum, list) else [argnum]
            variables = [args[i] for i in argnum_]
        for x in variables:
            assert isinstance(x, NDArray), "type of autograd input should NDArray."
        grads = [zeros_like(x) for x in variables]
        mark_variables(variables, grads)
        set_recording(True)
        outputs = func(*args)
        set_recording(False)
        compute_gradient([outputs] if isinstance(outputs, NDArray) else outputs)
        return grads, outputs
    return wrapped

def grad(func, argnum=None):
    """Return function that computes gradient of arguments.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.
    argnum: an int or a list of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_func: a python function
        A function that would compute the gradient of arguments.

    Examples
    --------
    >>> # autograd supports dynamic graph which is changed
    >>> # every instance
    >>> def func(x):
    >>>     r = random.randint(0, 1)
    >>>     if r % 2:
    >>>         return x**2
    >>>     else:
    >>>         return x/3
    >>> # use `grad(func)` to get the gradient function
    >>> for x in range(10):
    >>>     grad_func = grad(func)
    >>>     inputs = nd.array([[1, 2, 3], [4, 5, 6]])
    >>>     grad_vals = grad_func(inputs)
    """
    grad_with_loss_func = grad_and_loss(func, argnum)
    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]
    return wrapped
