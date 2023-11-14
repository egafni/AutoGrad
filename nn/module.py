import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

from nn.value import Value, tanh, exp

INFERENCE = False


@contextmanager
def inference_mode():
    global INFERENCE
    INFERENCE = True
    yield INFERENCE
    INFERENCE = False


class Module:
    def parameters(self, ):
        return self._parameters


class BatchNorm(Module):
    """needs checking"""

    def __init__(self, ):
        raise NotImplementedError('i have not double checked this')
        self.running_mean = None
        self.running_std = None
        self._parameters = None

    def __call__(self, x):
        mean = x.mean(0, keepdims=True)
        std = x.std(0, keepdims=True)

        if INFERENCE or self.running_mean is None:
            mean = self.running_mean
            std = self.running_std
        else:
            self.running_mean = self.running_mean * .999 + mean * .01
            self.running_std = self.running_std * .999 + std * .01

        x = [x - mean / std]
        return x


@dataclass
class Neuron(Module):
    n_in: int

    def __post_init__(self):
        self.w = [Value(random.random() * 2 - 1, name='w') for _ in range(self.n_in)]
        self.b = Value(0)

    def __call__(self, x):
        return sum(x_ * w_ for x_, w_ in zip(x, self.w)) + self.b

    def parameters(self, ):
        return self.w + [self.b]


@dataclass
class Layer(Module):
    n_in: int
    n_out: int

    def __post_init__(self):
        self.neurons = [Neuron(self.n_in) for _ in range(self.n_out)]

    def parameters(self, ):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        assert len(x) == self.n_in
        return [n(x) for n in self.neurons]


@dataclass
class Relu(Module):
    def __init__(self):
        self._parameters = []

    def __call__(self, x):
        return [max(0, x_.data) * x_ for x_ in x]


@dataclass
class Tanh(Module):
    def __init__(self):
        self._parameters = []

    def __call__(self, x):
        return [tanh(x_) for x_ in x]


@dataclass
class Softmax(Module):
    def __init__(self):
        self._parameters = []

    def __call__(self, x):
        m = max(x_.data for x_ in x)
        denom = sum(exp(x_ - m) for x_ in x)
        return [exp(x_ - m) / denom for x_ in x]


@dataclass
class MLP(Module):
    n_in: int
    layer_sizes: List[int]
    act: Module

    def __post_init__(self):
        prev = self.n_in
        layers = []
        for n in self.layer_sizes:
            layers += [Layer(prev, n), self.act()]
            prev = n

        self.layers = layers[:-1]  # remove last activation

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        """
        Note this class expects X to be batches of data
        """
        for layer in self.layers:
            x = layer(x)
        return x
