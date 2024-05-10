import numpy as np

def clamp(x, a, b):
    return np.maximum(a, np.minimum(b, x))


a = -5
b = 5


def floor(x):
    return clamp(np.floor(x), a, b)


def sigmoid_floor(x, T=0):
    y = (
        np.exp(-T) * (x + 1)
        - 1
        + np.sum(
            [
                1 / (1 + np.exp(-T * (x - i))) - 0.5
                for i in range(int(np.min(x)), int(np.max(x)))
            ],
            axis=0,
        )
    )
    return clamp(y, a, b)

def ste(x):
    # return 1 if x in a and b, 0 otherwise
    return np.logical_and(x >= a, x <= b).astype(np.float32)

def sigmoid_estimator(x, T=0):
    y = np.exp(-T) + np.sum(
        [
            np.exp(-T * (x - i)) * T / (1 + np.exp(-T * (x - i))) ** 2
            for i in range(int(np.min(x)), int(np.max(x)))
        ],
        axis=0,
    )
    return y * ste(x)



