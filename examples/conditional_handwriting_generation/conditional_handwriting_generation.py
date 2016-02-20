from collections import OrderedDict
import numpy as np
import theano
from theano import tensor

from dagbldr.datasets import fetch_iamondb, list_iterator
from dagbldr.optimizers import adam, gradient_clipping
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import create_checkpoint_dict
from dagbldr.utils import TrainingLoop
from dagbldr.nodes import location_attention_gru_recurrent_layer
from dagbldr.nodes import shift_layer
from dagbldr.nodes import gru_recurrent_layer, masked_cost
from dagbldr.nodes import bernoulli_and_correlated_gaussian_mixture_layer
from dagbldr.nodes import bernoulli_and_correlated_log_gaussian_mixture_cost


def plot_scatter_iamondb_example(X, title=None, equal=True):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    down = np.where(X[:, 0] == 0)[0]
    up = np.where(X[:, 0] == 1)[0]
    ax.scatter(X[down, 1], X[down, 2], color="steelblue")
    ax.scatter(X[up, 1], X[up, 2], color="darkred")
    if equal:
        ax.set_aspect('equal')
    if title is not None:
        plt.title(title)
    plt.show()


def delta(x):
    return np.hstack((x[1:, 0][:, None], x[1:, 1:] - x[:-1, 1:]))


def undelta(x):
    agg = np.cumsum(x[:, 1:], axis=0)
    return np.hstack((x[:, 0][:, None], agg))


def logsumexp(x, axis=None):
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def BivariateGMM(y, mu, sigma, corr, coeff, binary, epsilon = 1e-5):
    """
    Bivariate gaussian mixture model negative log-likelihood
    Parameters
    ----------
    """
    n_dim = y.ndim
    shape_y = y.shape
    #y = y.reshape((-1, shape_y[-1]))
    #y = y.dimshuffle(0, 1, 'x')

    binary = binary[:, :, 0]
    corr = corr[:, :, 0]

    mu_1 = mu[:, :, 0]
    mu_2 = mu[:,:, 1]

    sigma_1 = sigma[:, :, 0]
    sigma_2 = sigma[:, :, 1]

    binary = (binary+epsilon)*(1-2*epsilon)

    c_b = tensor.xlogx.xlogy0(y[:, :, 0],  binary) + tensor.xlogx.xlogy0(1 - y[:, :, 0], 1 - binary)

    inner1 = (0.5 * tensor.log(1.-corr**2 + epsilon))
    inner1 += tensor.log(sigma_1) + tensor.log(sigma_2)
    inner1 += tensor.log(2. * np.pi)

    y1 = y[:, :, 1][:, :, None]
    y2 = y[:, :, 2][:, :, None]
    Z = (((y1 - mu_1)/sigma_1)**2) + (((y2 - mu_2) / sigma_2)**2)
    Z -= (2. * (corr * (y1 - mu_1)*(y2 - mu_2)) / (sigma_1 * sigma_2))
    inner2 = 0.5 * (1. / (1. - corr**2 + epsilon))
    cost = - (inner1 + (inner2 * Z))

    theano.printing.Print("coeff")(coeff.shape)
    theano.printing.Print("cost")(cost.shape)
    nll = -logsumexp(tensor.log(coeff) + cost, axis=n_dim-1)
    theano.printing.Print("c_b")(c_b.shape)
    theano.printing.Print("nll")(nll.shape)
    nll -= c_b
    return nll.reshape(shape_y[:-1], ndim = n_dim-1)


iamondb = fetch_iamondb()
X = iamondb["data"]
y = iamondb["target"]
vocabulary_size = vs = iamondb["vocabulary_size"]
train_end = int(.9 * len(X))

random_state = np.random.RandomState(1999)
minibatch_size = 50

X = np.array([x.astype(theano.config.floatX) for x in X])
y = np.array([yy.astype(theano.config.floatX) for yy in y])

train_itr = list_iterator([X, y], minibatch_size, axis=1, make_mask=True,
                          stop_index=train_end)
X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)
train_itr.reset()

datasets_list = [X_mb, X_mb_mask, y_mb, y_mb_mask]
names_list = ["X", "X_mask", "y", "y_mask"]
graph = OrderedDict()
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, names_list, graph,
    list_of_test_values=[X_mb, X_mb_mask, y_mb, y_mb_mask])

n_hid = 400
n_out = 2

scan_kwargs = {"truncate_gradient": 300}
h1, att_p = location_attention_gru_recurrent_layer(
         [X_sym], [y_sym], X_mask_sym, y_mask_sym, n_hid, graph, 'l1_att_rec',
         random_state=random_state, n_gaussians=10, scan_kwargs=scan_kwargs)
w1 = att_p[0]
# Without shifting the network learns a crappy form of copying
X_shift = X_sym[:-1]
h2 = gru_recurrent_layer([h1, w1, X_shift], X_mask_sym[:-1], n_hid, graph, 'l2_rec',
                         random_state=random_state, scan_kwargs=scan_kwargs)
rval = bernoulli_and_correlated_gaussian_mixture_layer(
    [X_shift, h1, h2], graph, 'hw', proj_dim=n_out, n_components=20,
    random_state=random_state)
binary, coeffs, mus, sigmas, corr = rval
cost = BivariateGMM(X_sym[1:], mus, sigmas, corr, coeffs, binary,
                    epsilon = 1e-5)
#cost = bernoulli_and_correlated_log_gaussian_mixture_cost(
#    binary, coeffs, mus, sigmas, corr, X_sym)
#cost = masked_cost(cost, X_mask_sym).sum(axis=0).mean()
theano.printing.Print("cost")(cost.shape)
theano.printing.Print("X_mask_sym")(X_mask_sym.shape)

cost = (cost * X_mask_sym[1:]).sum()
cost = cost / X_mask_sym.sum()  # Shrink the learning rate by ~the mean seq length
params, grads = get_params_and_grads(graph, cost)

learning_rate = 1E-4
opt = adam(params, learning_rate)
clipped_grads = gradient_clipping(grads, 10.)
updates = opt.updates(params, clipped_grads)

fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                               [cost],
                               updates=updates)


cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost])
predict_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                   [binary, coeffs, mus, sigmas, corr])
attention_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                     att_p)

valid_itr = list_iterator([X, y], minibatch_size, axis=1, make_mask=True,
                          start_index=train_end)

checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(fit_function, cost_function,
                  train_itr, valid_itr,
                  checkpoint_dict=checkpoint_dict,
                  list_of_train_output_names=["train_cost"],
                  valid_output_name="valid_cost",
                  valid_frequency=300,
                  n_epochs=1000)
epoch_results = TL.run()
