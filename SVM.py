import numpy as np
import cvxopt
from matplotlib import pyplot as plt


class SVM(object):
    def __init__(self, C=None):
        self.w = None
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
            print("Regularization parameter C={:f}".format(self.C))

    def fit(self, X, y):

        n, d = X.shape

        yX = y[:, None] * X
        XTyT = yX.transpose()

        # Solving quadratic problem
        # From documentation: cvxopt.solvers.qp(P, q[, G, h[, A, b[, solver[, initvals]]]])
        P = cvxopt.matrix(np.dot(yX, XTyT))  # Gram matrix: y x x_T y_T
        q = cvxopt.matrix(np.ones((n, 1)) * -1) 
        A = cvxopt.matrix(y, (1, n), tc="d") # y
        b = cvxopt.matrix(0.0) # 0
        if self.C is None:
            # Hard-margin SVM
            h = cvxopt.matrix(np.zeros((n, 1)))
            G = cvxopt.matrix(np.identity(n) * -1)
        else:  # in case of C != None Lagrangian multipliers are upper bounded by C, so we need to add the constraints G*alpha <= C
            # Soft-margin SVM
            h_1 = cvxopt.matrix(np.zeros((n, 1)))
            h_2 = cvxopt.matrix(np.ones((n, 1)) * self.C)
            h = cvxopt.matrix(np.vstack((h_1, h_2)))
            G_1 = cvxopt.matrix(np.identity(n) * -1)
            G_2 = cvxopt.matrix(np.identity(n))
            G = cvxopt.matrix(np.vstack((G_1, G_2)))

        print("Fitting SVM by solving the Lagrangian dual quadratic problem.")
        self.solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Alphas: Lagrangian multipliers
        alpha = np.ravel(self.solution["x"])

        # Support vectors have alpha > 0
        support_vectors_index = alpha > 1e-3
        print()
        print("Support vector indeces: {}".format(np.where(support_vectors_index)[0]))

        # Support vectors are x_i with non-zero Lagrange multiplier
        self.x_sv = X[support_vectors_index]
        self.y_sv = y[support_vectors_index]
        self.alpha_sv = alpha[support_vectors_index]
        print()
        print("{:d} support vectors out of {:d} points".format(len(self.alpha_sv), n))
        print()
        # Weights vector
        self.w = np.zeros(d)
        for i in range(len(self.alpha_sv)):
            self.w += self.alpha_sv[i] * self.y_sv[i] * self.x_sv[i]
        print("Hyperplane weights w: {}".format(self.w))

        # Bias w0
        self.w0 = 0.0
        for i in range(len(self.alpha_sv)):
            self.w0 += self.y_sv[i] - np.dot(self.w, self.x_sv[i])
        self.w0 /= len(self.alpha_sv)
        print("Hyperplane intercept w0: {}".format(self.w0))
        print(
            "Hyperplane equation: {0:.3f}*x_i_0 + {1:.3f}*x_i_1 + {2:.3f}".format(
                self.w[0], self.w[1], self.w0
            )
        )

    def evaluate(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.w0
        else:
            print(
                "No model found. Try to estimate a SVM linear classifier by first runnig SVM().fit(X, y) function."
            )

    def predict(self, X):
        return np.sign(self.evaluate(X))

    def f(self, x, w, b, c=0):
        return (-w[0] * x - b + c) / w[
            1
        ]  # From hyperplane equation, solving w.r.t. X_1

    def plot_hyperplane(self, ax, X):
        # w.x + w0 = 0 --- Hyperplane
        a0 = np.amin(X[:, 0])
        b0 = np.amax(X[:, 0])
        a1 = self.f(a0, self.w, self.w0)
        b1 = self.f(b0, self.w, self.w0)
        print(ax)
        ax.plot([a0, b0], [a1, b1], "k")

    def plot_margins(self, ax, X):
        a0 = np.amin(X[:, 0])
        b0 = np.amax(X[:, 0])
        # w.x + w0 = 1 --- Bottom margin
        a1 = self.f(a0, self.w, self.w0, 1)
        b1 = self.f(b0, self.w, self.w0, 1)
        ax.plot([a0, b0], [a1, b1], "k--")
        # w.x + w0 = -1 --- Top Margin
        a1 = self.f(a0, self.w, self.w0, -1)
        b1 = self.f(b0, self.w, self.w0, -1)
        ax.plot([a0, b0], [a1, b1], "k--")
