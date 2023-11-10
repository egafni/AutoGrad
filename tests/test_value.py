import math
from typing import Literal

import numpy as np
import torch

from nn.value import Value, exp, log


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


def test_exp():
    def a():
        x1 = torch.tensor(5., requires_grad=True)
        x2 = torch.tensor(3., requires_grad=True)
        x3 = torch.exp(x1 + x2)
        x3 = x3*4
        x3.backward()
        return x1.grad.item(), x2.grad.item()

    def b():
        x1 = Value(5.)
        x2 = Value(3.)
        x3 = exp(x1 + x2)
        x3 = x3*4
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
        y = x1 * w1 + x2 * w2 - x1 ** 2 + w1 * w2 + exp_(x1) + log_(x1)
        # y = log_(y) + exp_(y)  # this breaks things!

        # zero
        for p in parameters:
            p.grad = 0.0 if backend == 'Value' else torch.tensor(0.0)

        y.backward()

        return x1.grad, x2.grad, w1.grad, w2.grad

    x1, x2 = 1., -1.
    w1, w2 = .1, .04

    assert np.isclose(_get_grads(x1, x2, w1, w2, 'Value'), _get_grads(x1, x2, w1, w2, 'torch')).all()
