import math
from typing import Literal

import numpy as np
import torch

from nn.value import Value, exp


def test_basics():
    x = Value(2.0)

    assert x + 1 == 3
    assert x - 1 == 1
    assert x ** 2 == 4
    assert x / 2 == 1.0

    assert 1 + x == 3
    assert 1 - x == -1
    assert 2 / x == 1

    assert exp(x).data == math.exp(x.data)


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
        elif backend == 'torch':
            f = torch.tensor
            exp_ = torch.exp
        else:
            raise ValueError(f'{backend} is invalid')

        x1, x2, w1, w2 = f(x1), f(x2), f(w1), f(w2)
        parameters = [x1, x2, w1, w2]

        for p in parameters:
            # only for torch
            p.requires_grad = True

        # forward
        y = x1 * w1 + x2 * w2 - x1**2 + w1*w2 + exp_(x1)

        # zero
        for p in parameters:
            p.grad = 0.0 if backend == 'Value' else torch.tensor(0.0)

        y.backward()

        return x1.grad, x2.grad, w1.grad, w2.grad

    x1, x2 = 5., -1.
    w1, w2 = .1, .04

    assert np.isclose(_get_grads(x1, x2, w1, w2, 'Value') , _get_grads(x1, x2, w1, w2, 'torch')).all()
