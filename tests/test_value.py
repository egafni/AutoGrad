import math
from typing import Literal

import numpy as np
import torch

from nn.module import Softmax
from nn.value import Value, exp, log, tanh


def test_basics():
    x = Value(2.0)

    assert x + 1 == 3
    assert x - 1 == 1
    assert x ** 2 == 4
    assert x / 2 == 1.0

    assert 1 + x == 3
    assert 1 - x == -1
    assert 2 / x == 1
    assert x ** -1 == 1 / 2

    assert exp(x).data == math.exp(x.data)
    assert log(x).data == math.log(x.data)


def test_softmax_tanh():
    # FIXME test grads
    x1 = [Value(2.0), Value(1.0), Value(-2.0), Value(40.)]
    x2 = torch.tensor([2.0, 1.0, -2, 40], requires_grad=True)

    assert (torch.tensor([tanh(x_).data for x_ in x1]) == torch.tanh(x2)).all()

    a1 = Softmax()(x1)

    a2 = torch.softmax(x2, dim=0)

    assert np.isclose([x.data for x in a1], a2.detach().numpy()).all()

    a1[0].backward()
    a2[0].backward()

    assert np.isclose(x1[0].grad, x2.grad[0]) and np.isclose(x1[1].grad, x2.grad[1])


def test_nll():
    y = [0, 1, ]
    logits = [[1, 2.0], [3, 4]]

    logits1 = [[Value(x[0]), Value(x[1])] for x in logits]
    logits2 = torch.tensor(logits, requires_grad=True)

    s1 = [Softmax()(x) for x in logits1]
    s2 = torch.softmax(logits2, dim=1)

    l1 = -torch.log(s2[range(len(s2)), y]).mean()
    l2 = -sum(log(x[y_]) for x, y_ in zip(s1, y)) / len(y)
    l1.backward()
    l2.backward()

    # grads should be the same, but they're not
    for i in range(len(logits)):
        for j in range(len(logits[i])):
            assert np.isclose(logits1[i][j].grad, logits2.grad[i][j])


def test_exp():
    def a():
        x1 = torch.tensor(5., requires_grad=True)
        x2 = torch.tensor(3., requires_grad=True)
        x3 = torch.exp(x1 + x2)
        x3 = x3 / 1000
        x3 = torch.exp(x3)
        x3.backward()
        return x1.grad.item(), x2.grad.item()

    def b():
        x1 = Value(5.)
        x2 = Value(3.)
        x3 = exp(x1 + x2)
        x3 = x3 / 1000
        x3 = exp(x3)
        x3.backward()
        return x1.grad, x2.grad

    assert np.isclose(a(), b()).all()


def test_backward():
    def _get_grads(x1: float, x2: float, w1: float, w2: float, backend: Literal['torch', 'Value']):
        """
        computes grads for y = x1 * w1 + x2 * w2 using `backend`
        used for comparing the torch to the Value backend

        :returns: x1.grad, x2.grad, w1.grad, w2.grad
        """
        if backend == 'Value':
            f = Value
            exp_ = exp
            log_ = log
        elif backend == 'torch':
            f = torch.tensor
            exp_ = torch.exp
            log_ = torch.log
        else:
            raise ValueError(f'{backend} is invalid')

        x1, x2, w1, w2 = f(x1), f(x2), f(w1), f(w2)
        parameters = [x1, x2, w1, w2]

        for p in parameters:
            # only for torch
            p.requires_grad = True

        # forward
        y = x1 * w1 + x2 * w2 - x1 ** 2 + w1 * w2 + exp_(x1) + log_(x1) / x1 + log_(x1) + x1 ** -3
        y = log_(y) + exp_(y)

        # zero
        for p in parameters:
            p.grad = 0.0 if backend == 'Value' else torch.tensor(0.0)

        y.backward()

        return x1.grad, x2.grad, w1.grad, w2.grad

    x1, x2 = 1., -1.
    w1, w2 = .1, .04

    assert np.isclose(_get_grads(x1, x2, w1, w2, 'Value'), _get_grads(x1, x2, w1, w2, 'torch')).all()
