import math
import numpy as np
from math import exp


def hardlim(n):
    """[Hard limit activation function]

    Args:
        n ([float]): [array]
    """
    n[n<0] = 0
    n[n>=0] = 1
    return n


def symmetrical_hardlimit(n):
    """[symmetrical_hardlimit]

    Args:
        n ([float]): [input activation]
    """
    n[n<0] = -1
    n[n>=0] = 1
    return n


def linear(n):
    """[Linear activation function]
  
    Args:
        n ([float]): [array]

    Returns:
        [float]: [return value input]
    """
    return n


def saturating_linear(n):
    """[activation function Saturating linear]

    Args:
        n ([float]): [array]

    Returns:
        [float]: [0 if n < 0, 1 if n > 1, n if 0<= n <= 1]
    """
    n[n<0] = 0
    n[n>1] = 1
    return n


def symmetrical_saturating_linear(n):
    """[summary]

    Args:
        n ([float]): [array]

    Returns:
        [float]: [array]
    """
    n[n <-1] = 0
    n[n > 1] = 1
    return n


def sigmoid(n):
    """[activation sigmoil: 1 / (1 + exp(-n))]

    Args:
        n ([float]): [array]

    Returns:
        [float]: [array]
    """
    return 1/(1+(np.exp((-n))))

def tanh(n):
    """[Hyperbolic tangent activation function]

    Args:
        n ([float]): [array]

    Returns:
        [float]: [description]
    """
    #((exp(n) - exp(-n))/(exp(n) + exp(-n)))
    return np.tanh(n)

def positive_linear(n):
    """[Positive linear activation function]

    Args:
        n ([float]): [array]

    Returns:
        [float]: [array]
    """
    n[n<0] = 0
    return n

def relu(x):
    """[summary]

    Args:
        x ([float]): [array]

    Returns:
        [float]: [array]
    """
    x[x<0]=0
    return x



def deserialize(name):
    globs = globals()
    obj = globs.get(name)
    if obj is None:
        raise ValueError(
            'Unknown activation name: {}.'.format(name)
        )
    return obj

def get(identifier):
    """[Returns function activation]

    Args:
        identifier : [Function or string]

    Raises:
        TypeError: [Input is an unknown function or string, i.e., the input does
        not denote any defined function]

    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        identifier = str(identifier)
        return deserialize(identifier)
    else:
        raise TypeError(
            'Could not interpret activation function identifier: {}'.format(
                identifier))

