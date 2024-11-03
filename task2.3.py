from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # for sigmoid function

class LogRegOracle(BaseSmoothOracle):
    """
    Oracle for logistic regression:
       func(x) = 1/m sum(log(1 + exp(-b_i * (a_i^T * x)))) + (C/2) * ||x||^2.
    """

    def __init__(self, A, b, regcoef=1):
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.C = regcoef

    def func(self, x):
        logits = self.A @ x
        return np.mean(np.log(1 + np.exp(-self.b * logits))) + (self.C / 2) * np.dot(x, x)

    def grad(self, x):
        logits = self.A @ x
        probabilities = expit(self.b * logits)
        gradient = -np.mean((self.b * (1 - probabilities))[:, np.newaxis] * self.A, axis=0)
        return gradient + self.C * x

    def hess(self, x):
        logits = self.A @ x
        probabilities = expit(self.b * logits)
        R = np.diag(probabilities * (1 - probabilities))
        hessian = (self.A.T @ R @ self.A) / self.m + self.C * np.eye(self.A.shape[1])
        return hessian

    def get_opt(self):
        LR = LogisticRegression(fit_intercept=False, C=1. / (self.C * self.m))
        LR.fit(self.A, (self.b + 1) / 2)
        return LR.coef_[0]
