import numpy as np
from IPython.display import Image
from graphviz import Digraph

def graph_parse(func):
        def tensor_check_and_operate(*args, **kwargs):
            tensors = [isinstance(arg, Tensor) for arg in args] #figure out which of the arguments are elements
            args = [args[i] if val else Tensor(args[i],name= 'num', track_grad = False) for i, val in enumerate(tensors)]
            values = tuple([args[i].value for i in range(len(args))]) #get the values of each argument
            #print(values)
            #print([i for i, val in enumerate(elements) if val])
            out = Tensor(func(*values, **kwargs), args, func.__name__) #create element with the output value of function and correct parents and creator function
            #print(out.parents)
            return out
        return tensor_check_and_operate

class Tensor(object):
    """A class representing an element of the computation graph
       with value."""


    def __init__(self, value, parents=None, function='NA', name = '', track_grad = True):
        if isinstance(value, list):
            self.value = np.array(value)
        elif isinstance(value, np.ndarray):
            self.value = value
        else:
            self.value = np.array([value])
        self.value = np.array(value)
        self.parents = parents
        self.function = function
        self.grad = 0
        self.name = name
        self.track_grad = track_grad

    @graph_parse
    def __add__(self, x):
        return self+x

    @graph_parse
    def __radd__(self,x):
        return self+x

    @graph_parse
    def __mul__(self, x):
        return self*x
    @graph_parse
    def __rmul__(self,x):
        return self*x

    @graph_parse
    def __truediv__(self,x):
        return self/x

    @graph_parse
    def __sub__(self, x):
        return self-x

    @graph_parse
    def __pow__(self,x):
        return self ** x

    def reshape(self,dims):
        self.value.reshape(dims)

    def __str__(self):
        return str("Tensor(shape: " + str(self.value.shape)
        + ', val: ' + str(self.value)
        +', grad: '+(str(self.grad.value) if isinstance(self.grad, Tensor) else str(self.grad))
        +', function: '+str(self.function))

    def backprop(self, grad = None):
        if grad is None:
            grad = Tensor(1)
        self.grad = (grad + self.grad) if isinstance(self.grad, Tensor) else grad
        self.grad.name = grad.name

        f_name = self.function if self.function is not None else None
        if self.parents is not None:
            for i, p in enumerate(self.parents):
                if p.track_grad:
                    if f_name == '__add__':
                        #backprop(p, grad = self.grad)
                        parent_grad = self.grad
                    elif f_name == '__mul__' or f_name =='__rmul__':
                        #backprop(p, grad = self.grad*(self.parents[1-i]))
                        parent_grad = self.grad*(self.parents[1-i])
                    elif f_name == '__sub__':
                        if i == 0:
                            #backprop(p, grad = self.grad)
                            parent_grad = self.grad
                        else:
                            #backprop(p, grad = self.grad*(-1))
                            parent_grad = self.grad*(-1)
                    elif f_name == '__truediv__':
                        if i == 0:
                            #backprop(p, grad = self.grad*(Element(1,None, None)/self.parents[1]))
                            parent_grad = self.grad*(Tensor(1)/self.parents[1])
                        else:
                            #backprop(p, grad = self.grad*(-1)*self.parents[0]*p**-2)
                            parent_grad = self.grad*(-1)*self.parents[0]*p**-2
                    elif f_name == '__pow__':
                        if i == 0:
                            #backprop(p, grad = self.grad*(self.parents[1])*p**(self.parents[1]-1))
                            parent_grad = self.grad*(self.parents[1])*p**(self.parents[1]-1)
                        else:
                            #backprop(p, grad = self.grad*0)
                            parent_grad = self.grad*(self.parents[0]**p)*log(self.parents[0],np.e)

                    elif f_name == 'log':
                        if i == 0:
                            parent_grad = self.grad*(Tensor(1)/p)*log(self.parents[1],np.e)
                        else:
                            parent_grad = self.grad*(-1)*(log(self.parents[0],np.e)/(p*np.log(p,np.e)**2))

                    elif f_name == 'sin':
                        parent_grad = self.grad*(cos(p))
                    elif f_name == 'cos':
                        parent_grad = self.grad*(sin(p)*(-1))
                    elif f_name == 'tan':
                        parent_grad = self.grad*(Tensor(1)/(cos(p)**2))

                    elif f_name =='matmul':
                        if i == 0:
                            parent_grad = matmul(self.grad, transpose(self.parents[1]))
                        else:
                            parent_grad = matmul(transpose(self.parents[0]),self.grad)

                    elif f_name == 'transpose':
                        parent_grad = transpose(self.grad)

                    elif f_name == 'hadamard':
                        parent_grad = hadamard(self.parents[1-i], self.grad)

                    elif f_name == 'reduce_sum':
                        parent_grad = self.grad * np.ones_like(p.value)

                    parent_grad.name = p.name+'grad'
                    #print(parent_grad)
                    p.backprop(grad = parent_grad)

    def traceback(self, first = True):

        l = [0,[]]
        l[0] = (self.name, self.value, self.grad.value if isinstance(self.grad, Tensor) else self.grad, self.function)
        if self.parents is not None:
            for parent in self.parents:
                t = traceback(parent, first = False)
                l[1].append(t[0])
                l[1].append(t[1])

        return l

    def grad_str(self):
        return str(self.grad.value) if isinstance(self.grad, Tensor) else str(self.grad)

    def graph_str(self):
        return (((self.name+'\n') if self.name is not '' else '')
        +'val: '+str(self.value)
        +'\ngrad: '+self.grad_str()
        +'\nfunc: '+self.function)

    def graph_visualize(self, graph = None, counter=0):

        if graph is None:
            graph = Digraph('graph')
            graph.node('0',self.graph_str())

        my_count = counter
        if self.parents is not None:
            for parent in self.parents:
                counter += 1
                graph.node(str(counter), parent.graph_str())
                graph.edge(str(my_count), str(counter))
                graph,counter = parent.graph_visualize(graph, counter = counter)

        if(my_count is not 0):
            return graph,counter
        else:
            return graph

    def flush_grads(self):
        self.grad = 0
        if self.parents is not None:
            for parent in self.parents:
                parent.flush_grads()

@graph_parse
def log(num, base):
    assert base is not 1
    return (np.log(num)/np.log(base))

@graph_parse
def sin(theta):
    return np.sin(theta)

@graph_parse
def cos(theta):
    return np.cos(theta)

@graph_parse
def tan(theta):
    return np.tan(theta)

@graph_parse
def matmul(a,b):
    return np.matmul(a,b)

@graph_parse
def hadamard(a,b):
    return a*b

@graph_parse
def transpose(a):
    return np.transpose(a)

@graph_parse
def reduce_sum(a):
    return np.sum(a)






def graph_print(l, name = False, value = True, grad = False, function = True):

    while True:
        isList = [isinstance(l[i], list) for i in range(len(l))]
        if False in isList:
            idxs = [i for i, val in enumerate(isList) if not val]
            idxs.reverse()
            to_print = []
            for i in idxs:
                info = l.pop(i)
                to_print.append((('n: %6s ' % info[0]) if name else '') +
                                (('val: %5.3f ' %info[1]) if value else '') +
                                (('g: %5.3f ' %info[2]) if grad else '')+
                                (('fun: %4s' % info[3].replace('_','')) if function else ''))
            to_print.reverse()
            print(to_print)

        elif l == []:
            break
        else:
            t = []
            for i in range(len(l)):
                for j in range(len(l[i])):
                    t.append(l[i][j])
            l = t
