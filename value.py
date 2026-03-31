class Value:

    def __init__(self, data: int, children=()):
        self.data = data
        self.grad = 0.0
        self.prev = children
        self.backward = lambda: None  # default none backward

    def __add__(self, other):
        if isinstance(other, int):
            other = Value(other)
        
        out = Value(self.data + other.data, (self, other,))

        def _backward():
            # already have out.grad here?
            self.grad = out.grad * 1
            other.grad = out.grad * 1
        out.backward = _backward

        return out
    
    def __mul__(self, other):
        if isinstance(other, int):
            other = Value(other)
        out = Value(self.data * other.data, children=(self, other))

        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        out.backward = _backward

        return out

    def __neg__(self, other):
        return self * -1

    def __sub__(self, other):
        return self.__add__(-other)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
if __name__ == '__main__':
    v1 = Value(2.0)
    v2 = Value(3.0)
    v3 = v1 * v2
    v3.grad = 1.0
    print(v3)
    v3.backward()
    print(v3)
