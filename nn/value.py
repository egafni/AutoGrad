import math
from typing import List


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
        return -self + other

    def __radd__(self, other):
        """other + self"""
        return self + other

    def __rmul__(self, other):
        """other * self"""
        return self * other

    def __rtruediv__(self, other):
        """other / self"""
        return other * self ** -1

    def topo_sort(self):
        topo_sorted = []
        seen = set()

        def _topo_sort(v):
            if v not in seen:
                seen.add(v)
                topo_sorted.append(v)
                for child in v._children:
                    _topo_sort(child)

        _topo_sort(self)
        return topo_sorted

    def backward(self):
        topo_sorted = self.topo_sort()
        # topo = []
        # visited = set()
        #
        # def build_topo(v):
        #     if v not in visited:
        #         visited.add(v)
        #         for child in v._children:
        #             build_topo(child)
        #         topo.append(v)
        #
        # build_topo(self)


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


def exp(x: Value):
    raise NotImplementedError('Unknown bugs occur proporgated gradients when there is nested composition of this method')
    v = Value(math.exp(x.data), _children=(x,))

    def _backward():
        x.grad += math.exp(x.data) * v.grad

    v._backward = _backward

    return v


def log(x: Value):
    raise NotImplementedError('Unknown bugs occur proporgated gradients when there is nested composition of this method')
    """log(x)"""
    v = Value(math.log(x.data), _children=(x,))

    def _backward():
        x.grad += 1 / x.data * math.log(math.e) * v.grad

    v._backward = _backward

    return v


def softmax(x: List[Value]):
    return [exp(x_) / sum(exp(x__) for x__ in x) for x_ in x]
