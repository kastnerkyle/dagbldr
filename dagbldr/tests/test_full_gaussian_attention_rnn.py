#!/usr/bin/env python
import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import softmax, linear, categorical_crossentropy
from dagbldr.nodes import gru
from dagbldr.nodes import gru_fork
from dagbldr.nodes import gaussian_attention
from dagbldr.nodes import masked_cost

from dagbldr import get_params

from dagbldr.optimizers import adam
from dagbldr.optimizers import gradient_norm_rescaling

def test_full_gaussian_attention_rnn():
    input_str = "abcdac"
    mult = 4
    s = list(set(input_str))
    indices = {k: v for k, v in zip(s, range(len(s)))}
    r_indices = {v: k for k, v in indices.items()}
    output_str = "".join([i * mult for i in input_str])

    def oh(arr):
        max_idx = max(arr) + 1
        o_arr = np.zeros((len(arr), max_idx)).astype("float32")
        for n, a in enumerate(arr):
            o_arr[n, a] = 1.
        return o_arr

    input_idx = np.array([indices[i] for i in input_str])
    output_idx = np.array([indices[o] for o in output_str])

    input_oh = oh(input_idx)
    output_oh = oh(output_idx)

    minibatch_size = 10
    input_oh = np.concatenate([input_oh[:, None, :] for _ in range(minibatch_size)], axis=1)
    output_oh = np.concatenate([output_oh[:, None, :] for _ in range(minibatch_size)], axis=1)

    input_mask = np.ones_like(input_oh[:, :, 0])
    output_mask = np.ones_like(output_oh[:, :,0])

    X_mb, y_mb, X_mb_mask, y_mb_mask = input_oh, output_oh, input_mask, output_mask

    y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
    y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))

    n_hid = 10
    n_ins = X_mb.shape[-1]
    n_outs = y_mb.shape[-1]
    n_ctx_ins = 2 * n_hid
    att_dim = 1

    train_enc_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
    train_enc_h1_r_init = np.zeros((minibatch_size, n_hid)).astype("float32")

    train_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
    train_w1_init = np.zeros((minibatch_size, n_ctx_ins)).astype("float32")
    train_k1_init = np.zeros((minibatch_size, att_dim)).astype("float32")

    X_sym = tensor.tensor3()
    y_sym = tensor.tensor3()
    X_mask_sym = tensor.fmatrix()
    y_mask_sym = tensor.fmatrix()

    enc_h1_0 = tensor.fmatrix()
    enc_h1_r_0 = tensor.fmatrix()
    h1_0 = tensor.fmatrix()

    w1_0 = tensor.fmatrix()
    w1_0.tag.test_value = train_w1_init

    k1_0 = tensor.fmatrix()
    k1_0.tag.test_value = train_k1_init

    X_sym.tag.test_value = X_mb
    X_mask_sym.tag.test_value = X_mb_mask
    y_sym.tag.test_value = y_mb
    y_mask_sym.tag.test_value = y_mb_mask

    enc_h1_0.tag.test_value = train_enc_h1_init
    enc_h1_r_0.tag.test_value = train_enc_h1_r_init
    h1_0.tag.test_value = train_h1_init

    y_tm1_sym = y_sym[:-1]
    y_tm1_mask_sym = y_mask_sym[:-1]

    y_t_sym = y_sym[1:]
    y_t_mask_sym = y_mask_sym[1:]

    random_state =np.random.RandomState(1999)
    init = "normal"

    def encoder_step(in_t, mask_t, in_r_t, mask_r_t, h1_tm1, h1_r_tm1):
        enc_h1_fork = gru_fork([in_t], [n_hid], n_hid,
                            name="enc_h1_fork",
                            random_state=random_state, init_func=init)
        enc_h1_t = gru(enc_h1_fork, h1_tm1, [n_hid], n_hid, mask=mask_t, name="enc_h1",
                    random_state=random_state, init_func=init)

        enc_h1_r_fork = gru_fork([in_r_t], [n_hid], n_hid,
                        name="enc_h1_r_fork",
                        random_state=random_state, init_func=init)
        enc_h1_r_t = gru(enc_h1_r_fork, h1_r_tm1, [n_hid], n_hid, mask=mask_r_t, name="enc_h1_r",
                random_state=random_state, init_func=init)
        return enc_h1_t, enc_h1_r_t

    X_proj = linear([X_sym], [n_ins], n_hid, init_func="unit_normal",
                    random_state=random_state)

    (enc_h1, enc_h1_r), _ = theano.scan(encoder_step,
                            sequences=[X_proj, X_mask_sym, X_proj[::-1], X_mask_sym[::-1]],
                            outputs_info=[enc_h1_0, enc_h1_r_0])

    enc_ctx = tensor.concatenate((enc_h1, enc_h1_r[::-1]), axis=2)

#average_step = 1.
#min_step = .9 * average_step
#max_step = 1.25 * average_step
    def step(in_t, mask_t, h1_tm1, k_tm1, w_tm1, ctx, ctx_mask):
        h1_t, k1_t, w1_t = gaussian_attention([in_t], [n_outs],
                                            h1_tm1, k_tm1, w_tm1,
                                            ctx, n_ctx_ins, n_hid,
                                            att_dim=att_dim,
                                            #average_step=average_step,
                                            #min_step=min_step,
                                            #max_step=max_step,
                                            cell_type="gru",
                                            conditioning_mask=ctx_mask,
                                            step_mask=mask_t,
                                            name="rec_gauss_att",
                                            random_state=random_state)
        return h1_t, k1_t, w1_t

    (h1, k, w), _ = theano.scan(step,
                                sequences=[y_tm1_sym, y_tm1_mask_sym],
                                outputs_info=[h1_0, k1_0, w1_0],
                                non_sequences=[enc_ctx, X_mask_sym])
    comb = h1
    y_pred = softmax([comb], [n_hid], n_outs, name="pred_l",
                    random_state=random_state, init_func=init)

    cost = categorical_crossentropy(y_pred, y_t_sym)
    loss = masked_cost(cost, y_t_mask_sym)
# normalize
    cost = loss.sum() / (y_mask_sym.sum() + 1.)

    params = list(get_params().values())
    grads = tensor.grad(cost, params)
    grads = gradient_norm_rescaling(grads, 5.)

    learning_rate = 0.0001
    opt = adam(params, learning_rate)
    updates = opt.updates(params, grads)

    in_args = [X_sym, y_sym, X_mask_sym, y_mask_sym]
    in_args += [enc_h1_0, enc_h1_r_0, h1_0, k1_0, w1_0]
    out_args = [cost, enc_h1, enc_h1_r, h1]
    pred_out_args = [y_pred, enc_h1, enc_h1_r, h1, k, w]

    fit_function = theano.function(in_args, out_args, updates=updates,
                                   mode="FAST_COMPILE")

    n_epochs = 1
    for i in range(n_epochs):
        #print("itr {}".format(i))
        cost, enc_h1, enc_h1_r, h1 = fit_function(X_mb, y_mb, X_mb_mask, y_mb_mask,
                                                train_enc_h1_init,
                                                train_enc_h1_r_init,
                                                train_h1_init,
                                                train_k1_init,
                                                train_w1_init)

    """
    predict_function = theano.function(in_args, pred_out_args)
    pred, enc_h1, enc_h1_r, h1_o, k_o, w_o = predict_function(X_mb, y_mb, X_mb_mask, y_mb_mask,
                                                            train_enc_h1_init,
                                                            train_enc_h1_r_init,
                                                            train_h1_init,
                                                            train_k1_init,
                                                            train_w1_init)
    best = pred.argmax(axis=-1)
    print("Expected output:")
    print(output_str)
    print("Got:")
    for ni in range(best.shape[1]):
        best_i = best[:, ni]
        str_ = "".join([r_indices[b_i] for b_i in best_i])
        print(str_)
    from IPython import embed; embed()
    raise ValueError()
    """
