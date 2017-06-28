#!/usr/bin/env python
import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import linear
from dagbldr.nodes import gru_fork
from dagbldr.nodes import vrnn
from dagbldr.nodes import gaussian_gaussian_log_kl

from dagbldr import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.optimizers import adam

from dagbldr.training import TrainingLoop
from dagbldr.datasets import minibatch_iterator


def make_sines(n_timesteps, n_offsets, harmonic=False, square=False):
    # Generate sinewaves offset in phase
    n_full = n_timesteps
    d1 = 3 * np.arange(n_full) / (2 * np.pi)
    d2 = 3 * np.arange(n_offsets) / (2 * np.pi)
    full_sines = np.sin(np.array([d1] * n_offsets).T + d2).astype("float32")
    # Uncomment to add harmonics
    if harmonic:
        full_sines += np.sin(np.array([1.7 * d1] * n_offsets).T + d2)
        full_sines += np.sin(np.array([7.362 * d1] * n_offsets).T + d2)
    if square:
        full_sines[full_sines <= 0] = 0
        full_sines[full_sines > 0] = 1
    full_sines = full_sines[:, :, None]
    return full_sines

n_timesteps = 50
minibatch_size = 2
full_sines = make_sines(10 * n_timesteps, minibatch_size, harmonic=False)
all_sines = full_sines[:n_timesteps]
n_full = 10 * n_timesteps
X = all_sines[:-1]
y = all_sines[1:]

n_in = 1
n_hid = 20
n_out = 1

train_itr = minibatch_iterator([X, y], minibatch_size, axis=1)
valid_itr = minibatch_iterator([X, y], minibatch_size, axis=1)

h_init = np.zeros((minibatch_size, n_hid)).astype("float32")

X_mb, y_mb = next(train_itr)
train_itr.reset()


X_sym = tensor.tensor3()
y_sym = tensor.tensor3()
h0 = tensor.fmatrix()
sample_flag = tensor.scalar()

X_sym.tag.test_value = X_mb
y_sym.tag.test_value = y_mb
sample_flag.tag.test_value = 1.
h0.tag.test_value = h_init

random_state = np.random.RandomState(1999)
X_fork = gru_fork([X_sym], [n_in], n_hid, name="h1",
                  random_state=random_state)

def step(in_t, h_tm1):
    h_t, phi_mu, phi_logsigma, prior_mu, prior_logsigma, z = vrnn(
            in_t, h_tm1, [n_hid], n_hid, sample_flag, name="rec",
            random_state=random_state)
    return h_t, phi_mu, phi_logsigma, prior_mu, prior_logsigma, z

r, updates = theano.scan(step,
                         sequences=[X_fork],
                         outputs_info=[h0, None, None, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

h = r[0]
phi_mu = r[1]
phi_logsigma = r[2]
prior_mu = r[3]
prior_logsigma = r[4]
sampled_z = r[5]

kld = gaussian_gaussian_log_kl([phi_mu], [phi_logsigma], [prior_mu], [prior_logsigma]).mean()

y_pred = linear([h], [n_hid], n_out, name="h2", random_state=random_state)

cost = ((y_sym - y_pred) ** 2).mean() - kld
params = list(get_params().values())
params = params
grads = tensor.grad(cost, params)

learning_rate = 0.0001
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym, h0, sample_flag], [cost, h], updates=updates)
cost_function = theano.function([X_sym, y_sym, h0, sample_flag], [cost, h])
predict_function = theano.function([X_sym, h0, sample_flag], [y_pred, h])

def train_loop(itr, extra_info):
    X_mb, y_mb = next(itr)
    cost, _ = fit_function(X_mb, y_mb, h_init, 1.)
    return [cost]


def valid_loop(itr, extra_info):
    X_mb, y_mb = next(itr)
    cost, _ = cost_function(X_mb, y_mb, h_init, 1.)
    return [cost]


checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=2000,
                  checkpoint_every_n_epochs=1000,
                  checkpoint_dict=checkpoint_dict,
                  skip_all_save=True)
epoch_results = TL.run()

# Run on self generations
n_seed = n_timesteps // 4
X_grow = X[:n_seed]
for i in range(n_timesteps // 4, n_full):
    p, _ = predict_function(X_grow, h_init, 0.)
    # take last prediction only
    X_grow = np.concatenate((X_grow, p[-1][None]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f, axarr1 = plt.subplots(minibatch_size, 3)
for i in range(minibatch_size):
    # -1 to have the same dims
    axarr1[i, 0].plot(full_sines[:-1, i, 0], color="steelblue")
    axarr1[i, 1].plot(X_grow[:, i, 0], color="darkred")
    axarr1[i, 2].plot(np.abs(X_grow[:-1, i, 0] - full_sines[:-1, i, 0]),
                      color="darkgreen")
plt.savefig('out.png')
