class LinRegOracle(BaseSmoothOracle):
    """
    Oracle for linear regression:
       func(x) = 1/m ||Ax - b||^2.
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.m = A.shape[0]

    def func(self, x):
        residuals = self.A @ x - self.b
        return np.sum(residuals ** 2) / self.m

    def grad(self, x):
        return 2 * self.A.T @ (self.A @ x - self.b) / self.m

    def hess(self, x):
        return 2 * self.A.T @ self.A / self.m

    def get_opt(self):
        return np.linalg.solve(self.A.T @ self.A, self.A.T @ self.b)
