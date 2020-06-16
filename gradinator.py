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
            out = Tensor(func(*values, **kwargs), parents = args, function =  func.__name__) #create element with the output value of function and correct parents and creator function
            #for p in args:
            #    p.dependents.add(out)
            #print(out.parents)
            return out
        return tensor_check_and_operate

class Graph_track(object):

    def __init__(self):
        self.d_list = []

    def add(self, t):
        self.d_list.append(t)
        return

    def insert(self, node):
        if not node in self.d_list:
            l = len(self.d_list)
            if l == 0:
                self.add(node)
            else:
                for i in reversed(range(l)):
                    if node.depth>=self.d_list[i].depth:
                        self.d_list.insert(i+1, node)
                        return
                self.d_list.insert(0,node)
                return


    def get_next(self):
        l = len(self.d_list)
        if l == 0:
            return None
        n = self.d_list[0]
        if l>0:
            self.d_list = self.d_list[1:]
        return n

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
        self.depth = 0
        #self.dependents = set()

    def is_scalar(self):
        #a hacky solution to the scalar issue, will not work with vectorized samples
        return self.value.shape == ()

    @graph_parse
    def __add__(self, x):
        return self+x

    def __radd__(self,x):
        return self+x

    @graph_parse
    def __mul__(self, x):
        return self*x

    def __rmul__(self,x):
        return self*x

    def __neg__(self):
        return (-1)*self

    @graph_parse
    def __truediv__(self,x):
        return self/x

    @graph_parse
    def __sub__(self, x):
        return self-x

    def __rsub__(self,x):
        return (-1)*(self-x)
    @graph_parse
    def __pow__(self,x):
        return self ** x

    def reshape(self,dims):
        self.value.reshape(dims)

    def __str__(self):
        return str("Tensor(shape: " + str(self.value.shape)
        + ', val: ' + str(self.value)
        +', grad: '+(str(self.grad.value) if isinstance(self.grad, Tensor) else str(self.grad))
        +', function: '+str(self.function)
        +', name: '+ self.name)

    def scalar_check(self,p,i,parent_grad):
        if p.is_scalar() and not self.parents[1-i].is_scalar():
            return reduce_sum(parent_grad)
        return parent_grad

    def back(self, grad = None, tracker= None, first = True):
        if grad is None:
            grad = Tensor(1)

        if tracker is None:
            tracker = Graph_track()


        self.grad = (grad + self.grad) if isinstance(self.grad, Tensor) else grad
        self.grad.name = grad.name

        #if len(self.dependents) is not 0:
        #    tracker.add(self)
            #print('adding: '+str(self))
            #print(self.dependents)
        #    return

        f_name = self.function if self.function is not None else None

        p_grad_list = []
        if self.parents is not None:
            for i, p in enumerate(self.parents):
                if p.track_grad:
                    if f_name == '__add__':
                        #backprop(p, grad = self.grad)
                        parent_grad = self.grad
                        parent_grad = self.scalar_check(p,i,parent_grad)
                    elif f_name == '__mul__' or f_name =='__rmul__':
                        #backprop(p, grad = self.grad*(self.parents[1-i]))
                        parent_grad = self.grad*(self.parents[1-i])

                        parent_grad = self.scalar_check(p,i,parent_grad)

                    elif f_name == '__sub__':
                        if i == 0:
                            #backprop(p, grad = self.grad)
                            parent_grad = self.grad
                        else:
                            #backprop(p, grad = self.grad*(-1))
                            parent_grad = self.grad*(-1)
                        parent_grad = self.scalar_check(p,i,parent_grad)
                    elif f_name == '__truediv__':
                        if i == 0:
                            #backprop(p, grad = self.grad*(Element(1,None, None)/self.parents[1]))
                            parent_grad = self.grad*(Tensor(1)/self.parents[1])
                        else:
                            #backprop(p, grad = self.grad*(-1)*self.parents[0]*p**-2)
                            parent_grad = self.grad*(-1)*self.parents[0]*p**-2

                        parent_grad = self.scalar_check(p,i,parent_grad)

                    elif f_name == '__pow__':
                        if i == 0:
                            #backprop(p, grad = self.grad*(self.parents[1])*p**(self.parents[1]-1))
                            parent_grad = self.grad*(self.parents[1])*p**(self.parents[1]-1)
                        else:
                            #backprop(p, grad = self.grad*0)
                            parent_grad = self.grad*(self.parents[0]**p)*log(self.parents[0],np.e)

                        parent_grad = self.scalar_check(p,i,parent_grad)

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

                    elif f_name == 'relu':
                        parent_grad = self.grad * np.clip(p.value, a_min=0,a_max=None)/p.value

                    parent_grad.name = p.name+'grad'
                    #print(parent_grad)
                    #p.dependents.remove(self)
                    #print(str(p)+str(p.dependents))
                    p_grad_list.append(parent_grad)

            for i, p in enumerate(self.parents):
                if p.track_grad:
                    p.back(grad = p_grad_list[i], tracker = tracker, first = False)

        #if first:
        #    while not tracker.isEmpty():
        #        tracker.process()
        #        print([str(list(tracker.processing[i].dependents)[0].parents[0]) for i in range(len(tracker.processing))])

    def traceback(self, first = True):

        l = [0,[]]
        l[0] = (self.name, self.value, self.grad.value if isinstance(self.grad, Tensor) else self.grad, self.function)
        if self.parents is not None:
            for parent in self.parents:
                t = traceback(parent, first = False)
                l[1].append(t[0])
                l[1].append(t[1])

        return l

    def trace_depth(self, depth=0):
        if self.depth < depth:
            self.depth = depth
        if self.parents is not None:
            for parent in self.parents:
                parent.trace_depth(depth+1)


    def list_order(self, tracker: Graph_track):
        #Assumes that all depths have been calculated
        tracker.insert(self)
        if self.parents is not None:
            for p in self.parents:
                p.list_order(tracker)


    def update_parents(self):
        f_name = self.function if self.function is not None else None

        if self.parents is not None:
            for i, p in enumerate(self.parents):
                if p.track_grad:
                    if f_name == '__add__':
                        #backprop(p, grad = self.grad)
                        parent_grad = self.grad
                        parent_grad = self.scalar_check(p,i,parent_grad)
                    elif f_name == '__mul__' or f_name =='__rmul__':
                        #backprop(p, grad = self.grad*(self.parents[1-i]))
                        parent_grad = self.grad*(self.parents[1-i])

                        parent_grad = self.scalar_check(p,i,parent_grad)

                    elif f_name == '__sub__':
                        if i == 0:
                            #backprop(p, grad = self.grad)
                            parent_grad = self.grad
                        else:
                            #backprop(p, grad = self.grad*(-1))
                            parent_grad = self.grad*(-1)
                        parent_grad = self.scalar_check(p,i,parent_grad)
                    elif f_name == '__truediv__':
                        if i == 0:
                            #backprop(p, grad = self.grad*(Element(1,None, None)/self.parents[1]))
                            parent_grad = self.grad*(Tensor(1)/self.parents[1])
                        else:
                            #backprop(p, grad = self.grad*(-1)*self.parents[0]*p**-2)
                            parent_grad = self.grad*(-1)*self.parents[0]*p**-2

                        parent_grad = self.scalar_check(p,i,parent_grad)

                    elif f_name == '__pow__':
                        if i == 0:
                            #backprop(p, grad = self.grad*(self.parents[1])*p**(self.parents[1]-1))
                            parent_grad = self.grad*(self.parents[1])*p**(self.parents[1]-1)
                        else:
                            #backprop(p, grad = self.grad*0)
                            parent_grad = self.grad*(self.parents[0]**p)*log(self.parents[0],np.e)

                        parent_grad = self.scalar_check(p,i,parent_grad)

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

                    elif f_name == 'relu':
                        parent_grad = self.grad * np.clip(p.value, a_min=0,a_max=None)/p.value

                    parent_grad.name = p.name+'grad'
                    #print(parent_grad)
                    #p.dependents.remove(self)
                    #print(str(p)+str(p.dependents))
                    #p_grad_list.append(parent_grad)
                    #p.backprop(grad = p_grad_list[i], tracker = tracker, first = False)
                    p.grad = p.grad+parent_grad if isinstance(p.grad,Tensor) else parent_grad


    def backprop(self):
        self.trace_depth()
        tracker = Graph_track()
        self.list_order(tracker)
        self.grad = Tensor(1)
        n = tracker.get_next()
        while n is not None:
            n.update_parents()
            n = tracker.get_next()


    def grad_str(self):
        return str(self.grad.value) if isinstance(self.grad, Tensor) else str(self.grad)

    def graph_str(self, disp_val = True):
        return (((self.name+'\n') if self.name is not '' else '')
        +(('val: '+str(self.value)) if disp_val else ('shape: '+ str(self.value.shape)))
        +'\ngrad: '+self.grad_str()
        +'\nfunc: '+self.function)

    def graph_visualize(self, graph = None, counter=0, disp_val = True):

        if graph is None:
            graph = Digraph('graph')
            graph.node('0',self.graph_str(disp_val = disp_val))

        my_count = counter
        if self.parents is not None:
            for parent in self.parents:
                counter += 1
                graph.node(str(counter), parent.graph_str(disp_val = disp_val))
                graph.edge(str(my_count), str(counter))
                graph,counter = parent.graph_visualize(graph, counter = counter,disp_val=disp_val)

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

def ln(num):
    return log(num, np.e)

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

@graph_parse
def relu(x):
    return np.clip(x,a_min=0,a_max=None)





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
