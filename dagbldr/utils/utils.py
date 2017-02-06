# Author: Kyle Kastner
# License: BSD 3-clause
import re
import numpy as np
import theano
from theano import tensor
from collections import OrderedDict

from ..core import safe_zip
from ..core import get_type
from ..core import get_lib_shared_params

_type = get_type()


def numpy_one_hot(labels_dense, n_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dtype = labels_dense.dtype
    labels_dense = labels_dense.ravel().astype("int32")
    n_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((n_labels, n_classes))
    labels_one_hot[np.arange(n_labels).astype("int32"),
                   labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(n_classes,))
    return labels_one_hot.astype(labels_dtype)


def get_weights(accept_regex="_W", skip_regex="_softmax_"):
    """
    A regex matcher to get weights. To bypass, simply pass None.

    Returns dictionary of {name: param}
    """
    d = get_lib_shared_params()
    if accept_regex is not None:
        ma = re.compile(accept_regex)
    else:
        ma = None
    if skip_regex is not None:
        sk = re.compile(skip_regex)
    else:
        sk = None
    matched_keys = []
    for k in d.keys():
        if ma is not None:
            if ma.search(k):
                if sk is not None:
                    if not sk.search(k):
                        matched_keys.append(k)
                else:
                    matched_keys.append(k)
    matched_weights = OrderedDict()
    for mk in matched_keys:
        matched_weights[mk] = d[mk]
    return matched_weights


def as_shared(arr, **kwargs):
    return theano.shared(np.cast[_type](arr))


def concatenate(tensor_list, axis=0):
    """
    Wrapper to `theano.tensor.concatenate`, that casts everything to float32!
    """
    out = tensor.cast(tensor.concatenate(tensor_list, axis=axis),
                      dtype=_type)
    # Temporarily commenting out - remove when writing tests
    # conc_dim = int(sum([calc_expected_dim(graph, inp)
    #                for inp in tensor_list]))
    # This may be hosed... need to figure out how to generalize
    # shape = list(expression_shape(tensor_list[0]))
    # shape[axis] = conc_dim
    # new_shape = tuple(shape)
    # tag_expression(out, name, new_shape)
    return out


def interpolate_between_points(arr, n_steps=50):
    """ Helper function for drawing line between points in space """
    assert len(arr) > 2
    assert n_steps > 1
    path = [path_between_points(start, stop, n_steps=n_steps)
            for start, stop in safe_zip(arr[:-1], arr[1:])]
    path = np.vstack(path)
    return path


def path_between_points(start, stop, n_steps=100, dtype=theano.config.floatX):
    """ Helper function for making a line between points in ND space """
    assert n_steps > 1
    step_vector = 1. / (n_steps - 1) * (stop - start)
    steps = np.arange(0, n_steps)[:, None] * np.ones((n_steps, len(stop)))
    steps = steps * step_vector + start
    return steps.astype(dtype)
