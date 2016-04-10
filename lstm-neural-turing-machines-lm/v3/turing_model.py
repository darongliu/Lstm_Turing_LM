import theano
import theano.tensor as T
import controller
import model
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P , input_size , mem_width , weighted_mem_width , output_size) :
    ctrl = controller.build(P, input_size, output_size, weighted_mem_width)
    predict = model.build(P, input_size, mem_width, weighted_mem_width, ctrl)

    def turing_updates(cost , lr) :
        params = P.values()
        #whether add P weight decay
        l2 = T.sum(0)
        for p in params:
            l2 = l2 + (p ** 2).sum()
        all_cost = cost + 1e-3 * l2 
        grads = [T.clip(g, -100, 100) for g in T.grad(all_cost, wrt=params)]
        return updates.rmsprop(params, grads, learning_rate=lr)
    
    def init_parameter(name , value) :
        P[name] = value #used by getvalue

    return turing_updates , predict


        
