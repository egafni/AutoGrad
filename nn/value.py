class Value:
    def __init__(self, data, _children=(), name=None):
        assert isinstance(data, (float, int))

        self.data = data
        self._children = set(_children)
        self.name = name

        self._backward = lambda: None
        self.grad = 0.0

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        v = Value(self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += 1 * v.grad
            other.grad += 1 * v.grad

        v._backward = _backward

        return v

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        v = Value(self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += other.data * v.grad
            other.grad += self.data * v.grad

        v._backward = _backward

        return v

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        v = Value(self.data ** other, _children=(self,))

        def _backward():
            self.grad += (other * self.data) ** (other - 1) * v.grad

        v._backward = _backward

        return v

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        """self / other"""
        return self * other ** -1

    def __sub__(self, other):
        """self - other"""
        return self + -other

    def __rsub__(self, other):
        """other - self"""
        return other + -self

    def __rtruediv__(self, other):
        """other / self"""
        return other * self ** -1

    def backward(self):
        topo_sorted = []
        seen = set()

        def topo_sort(v):
            if v not in seen:
                seen.add(v)
                topo_sorted.append(v)
                for child in v._children:
                    topo_sort(child)

        topo_sort(self)

        self.grad = 1.0
        for v in topo_sorted:
            v._backward()

    def __repr__(self):
        if self.name:
            return f'Value({self.name}={self.data}, grad={self.grad})'
        else:
            return f'Value({self.data}, grad={self.grad})'

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.data == other.data
        else:
            return self.data == other

    def __hash__(self):
        return hash(id(self))
