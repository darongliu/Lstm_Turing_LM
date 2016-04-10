import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U


def build(P, id, input_size, mem_width):

#    P["W_%d_key" % id] = U.initial_weights(input_size, mem_width)
    P["W_%d_key" % id] = U.initial_weights(input_size, mem_width)
    P["b_%d_key" % id] = 0. * U.initial_weights(mem_width)
#    P["W_%d_shift" % id] = U.initial_weights(input_size, shift_width)
#    P["b_%d_shift" % id] = 0. * U.initial_weights(shift_width)

    P["W_%d_beta" % id] = 0. * U.initial_weights(input_size)
    P["b_%d_beta" % id] = 0.
    P["W_%d_gamma" % id] = U.initial_weights(input_size)
    P["b_%d_gamma" % id] = 0.
    if id != 0 :
        P["W_%d_g" % id] = U.initial_weights(input_size)
        P["b_%d_g" % id] = 0.

    def head_params(x):
        # key
        key_t = T.dot(x, P["W_%d_key" % id]) + P["b_%d_key" % id]

        _beta_t = T.dot(x, P["W_%d_beta" % id]) + P["b_%d_beta" % id]#guess this is a vector
        _gamma_t = T.dot(x, P["W_%d_gamma" % id]) + P["b_%d_gamma" % id]#guess this is a vector

        beta_t = T.nnet.softplus(_beta_t)
        gamma_t = T.nnet.softplus(_gamma_t) + 1.
#		beta_t  = (_beta_t  > 0)*_beta_t
#		gamma_t = (_gamma_t > 0)*_gamma_t + 1.
#		beta_t  = T.exp(_beta_t)
#		gamma_t = T.exp(_gamma_t) + 1.
        if id != 0 :
            g_t = T.nnet.sigmoid(T.dot(x, P["W_%d_g" % id]) + P["b_%d_g" % id])#guess this is a vector        
        else :
            g_t = T.ones([x.shape[0]])

        return key_t, beta_t, g_t, gamma_t
    return head_params
