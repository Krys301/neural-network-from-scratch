class Value:
    def __init__(self,data,_children = (),_op =''):
        self.data = data 
        self.grad = 0.0 # gradient always starts at 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda :None # default does nothing 
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

a = Value(2.0)
b = Value(3.0)
c = Value(4.0)
d= a*b
e = d+c
e.backward()




print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
print(f"c.grad = {c.grad}")


