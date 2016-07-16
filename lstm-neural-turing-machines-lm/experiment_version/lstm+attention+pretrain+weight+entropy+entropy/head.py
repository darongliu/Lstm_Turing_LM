import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U


def build(P, id, input_size, mem_width):

#    P["W_%d_key" % id] = U.initial_weights(input_size, mem_width)
    P["W_%d_key" % id] = U.initial_weights(input_size, mem_width)
    P["b_%d_key" % id] = 0. * U.initial_weights(mem_width)
    P["W_%d_sim" % id] = U.initial_weights(input_size, mem_width)
    P["b_%d_sim" % id] = 0. * U.initial_weights(mem_width)
#    P["W_%d_shift" % id] = U.initial_weights(input_size, shift_width)
#    P["b_%d_shift" % id] = 0. * U.initial_weights(shift_width)

    if id != 0 :
        P["W_%d_g" % id] = U.initial_weights(input_size)
        P["b_%d_g" % id] = 0.

    def head_params(x):
        # key
        key_t = T.dot(x, P["W_%d_key" % id]) + P["b_%d_key" % id]
        
        #similarity weight
        sim_t = T.nnet.sigmoid(T.dot(x, P["W_%d_sim" % id]) + P["b_%d_sim" % id])
        #attention  weight
        att_t = sim_t

        
        if id != 0 :
            g_t = T.nnet.sigmoid(T.dot(x, P["W_%d_g" % id]) + P["b_%d_g" % id])#guess this is a vector        
        else :
            g_t = T.ones([x.shape[0]])

        return key_t, g_t, sim_t, att_t
    return head_params
