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
    P.W_read_hidden  = U.initial_weights(mem_width, output_size)
    P.b_hidden_0 = 0. * U.initial_weights(output_size)

    def controller(input_t, read_t):
        #		print "input_t",input_t.type
        if input_t.ndim > 1 :
            output_t = T.nnet.softmax(
                T.dot(input_t, P.W_input_hidden) +
                T.dot(read_t, P.W_read_hidden) +
                P.b_hidden_0
            )
        else :
            output_t = U.vector_softmax(
                T.dot(input_t, P.W_input_hidden) +
                T.dot(read_t, P.W_read_hidden) +
                P.b_hidden_0)

#		print "input",read_t.type,input_t.type
#		print "weights",P.W_input_hidden.type,P.W_read_hidden.type,P.b_hidden_0.type
#		print "layer", hidden_0.type

        return output_t
    return controller
