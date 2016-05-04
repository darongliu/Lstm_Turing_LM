import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano.tensor.extra_ops import repeat
import controller
import head
import scipy


def cosine_sim(k, M):
    k_lengths = T.sqrt(T.sum(k**2, axis=1)).dimshuffle((0, 'x'))
    k_unit = k / (k_lengths + 1e-5)
    k_unit.name = "k_unit"
    M_lengths = T.sqrt(T.sum(M**2, axis=2)).dimshuffle((0, 1, 'x'))
    M_unit = M / (M_lengths + 1e-5)
    M_unit.name = "M_unit"
    M_trans = M_unit.dimshuffle((0,2,1))
    return T.batched_dot(k_unit , M_trans)


def build_step(P, controller,
               mem_width,
               similarity=cosine_sim,
               no_heads=1):
#    shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[
#            np.arange(-(shift_width // 2), (shift_width // 2) + 1)
#        ][::-1]
    P.memory_init = 2 * (np.random.rand(1, mem_width) - 0.5).astype(theano.config.floatX)
    heads = [head.build(P, h, mem_width, mem_width) for h in range(no_heads)]

    def build_read(M_curr, weight_curr):
        return T.batched_dot(weight_curr, M_curr)

#    def shift_convolve(weight, shift):
#        return T.batched_dot(shift , weight[:,shift_conv])

    def build_head_curr(weight_prev, M_curr, head, input_curr):
        """
        This function is best described by Figure 2 in the paper.
        """
        key, g = head(input_curr)

        # 3.3.1 Focusing b Content
        weight_c = T.nnet.softmax(similarity(key, M_curr))
        weight_c.name = "weight_c"

        # 3.3.2 Focusing by Location
        weight_g = T.batched_dot(g , weight_c) + T.batched_dot((1 - g) , weight_prev)
        weight_g.name = "weight_g"

#        weight_shifted = shift_convolve(weight_g, shift)#not modified     

        return weight_g

    def step(time_idx,lstm_hidden):
        M_pad = repeat(P.memory_init.dimshuffle((0,'x',1)) , lstm_hidden.shape[1] , axis=1 )
        M_curr_temp = T.concatenate([M_pad , lstm_hidden[:time_idx,:,:]] , axis=0)
        M_curr      = M_curr_temp.transpose((1,0,2))
        input_curr  = lstm_hidden[time_idx,:,:]

        weight_prev = T.zeros([input_curr.shape[0] , time_idx+1])
        weight_inter = weight_prev

        for head in heads:
            weight_inter = build_head_curr(
                weight_inter, M_curr , head, input_curr)

        weight_curr = weight_inter

        read_curr = build_read(M_curr, weight_curr)
        output = controller(input_curr, read_curr)

        return output
    return step


def build(P, mem_width, ctrl):
    step = build_step(
        P, ctrl, mem_width)

    def predict(lstm_output_gate):
        time_idx = T.arange(lstm_output_gate.shape[0])
        #axis:
        #0:time sequence
        #1:batch
        #2:output dim

        outputs , _ = theano.scan(
            step,
            sequences=time_idx,
            outputs_info = None,
            non_sequences=lstm_output_gate
        )
        return outputs
    return predict
