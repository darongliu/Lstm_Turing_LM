import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

import controller
import model
import head
from collections import namedtuple
#from theano_toolkit.parameters import Parameters


def build(P, input_size, output_size, mem_width):
    """
    Create controller function for use during scan op
    """

    P.W_input_hidden = U.initial_weights(input_size, output_size)
    P.b_hidden_0 = 0. * U.initial_weights(output_size)
    P.attention_weight = np.array(0.1,dtype=theano.config.floatX)

    def controller(input_t, read_t):
        #		print "input_t",input_t.type
        lstm_weight = 1-P.attention_weight
        weighted_sum = lstm_weight*input_t + P.attention_weight*read_t

        if input_t.ndim > 1 :
            output_t = T.nnet.softmax(
                T.dot(weighted_sum, P.W_input_hidden) +
                P.b_hidden_0
            )
        else :
            output_t = U.vector_softmax(
                T.dot(weighted_sum, P.W_input_hidden) +
                P.b_hidden_0)

#		print "input",read_t.type,input_t.type
#		print "weights",P.W_input_hidden.type,P.W_read_hidden.type,P.b_hidden_0.type
#		print "layer", hidden_0.type

        return output_t
    return controller
