import numpy as np

class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(x, self.A @ x) - np.dot(self.b, x)

    def grad(self, x):
        return self.A @ x - self.b

    def hess(self, x):
        return self.A

    def get_opt(self):
        return np.linalg.solve(self.A, self.b)
