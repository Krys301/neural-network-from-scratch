import math 
import random 

class Value:
    def __init__(self,data,_children = (),_op =''):
        self.data = data 
        self.grad = 0.0 # gradient always starts at 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda :None # default does nothing 
    
    def tanh(self):
        t = math.tanh(self.data)
        result = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1-t**2)*result.grad 
        result._backward = _backward
        return result 
    
    def backward(self):
        topo = []
        visited =set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
        self.grad =1.0
        for v in reversed(topo):
            v._backward()
  
    def __add__(self,other):
        result = Value(self.data + other.data, (self,other),'+')
        def _backward():
            #gradient flows equally to both inputs for addition 
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward
        return result
   
    def __mul__(self,other):
        result = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward 
        return result
    
    def __repr__(self):
         return f"Value(data={self.data}, grad={self.grad})"
class Nueron:
    def __init__(self,num_inputs):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self,inputs):
        activation = sum((w*Value(x) for w,x in zip(self.weights,inputs)),self.bias)

        return activation.tanh()
    def parameters(self):
        return self.weights
n = Nueron(3)
inputs =[2.0,3.0,-1.0]
output = n(inputs)
print(f"neuron outputs: {output}")

output.backward()

print(f"gradients after backprop:")
for i,p in enumerate(n.parameters()):
    print(f"param {i}: data = {p.data:.4f}, grad = {p.grad:.4f}")

learning_rate = 0.01 
print("\nBefore update:")
for p in n.parameters():
    print(f"data = {p.data:.4f}, grad = {p.grad:.4f}")

for p in n.parameters():
    p.data -= learning_rate * p.grad

print("\nafter update: ")
for p in n.parameters():
    print(f"  data={p.data:.4f}, grad={p.grad:.4f}")