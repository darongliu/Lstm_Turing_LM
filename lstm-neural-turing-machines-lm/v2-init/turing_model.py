import theano
import theano.tensor as T
import controller
import model
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P , input_size , mem_width , output_size) :
    ctrl = controller.build(P, input_size, output_size, mem_width)
    predict = model.build(P, input_size, mem_width, ctrl)

    def turing_updates(cost , lr) :
        params = P.values()
        #whether add P weight decay
        l2 = T.sum(0).astype(theano.config.floatX)
        for p in params:
            l2 = l2 + (p ** 2).sum().astype(theano.config.floatX)
        all_cost = cost + 1e-3 * l2 
        clipper = updates.clip(5.)
        g = T.grad(all_cost, wrt=params)
        grads = clipper(g)
#        grads = [T.clip(g, -5, 5) for g in T.grad(all_cost, wrt=params)]
#        return updates.rmsprop(params, grads, learning_rate=lr)
        return updates.momentum(params, grads, mu = 0, learning_rate=lr)        
    
    def init_parameter(name , value) :
        P[name] = value #used by getvalue

    return turing_updates , predict


        
