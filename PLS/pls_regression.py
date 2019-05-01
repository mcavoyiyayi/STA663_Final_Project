import PLS.pls_1d as pls_1d
import PLS.pls_nd as pls_nd
import pyximport; pyximport.install()
import PLS.PLS_cython as pls_cy
import numpy as np


class pls_regression:
    def __init__(self, cythoinzed = True):
        self.W = {}
        self.B = {}
        self.P = {}
        self.Q = {}
        self.dim = -1
        self.rank = 0
        self.cythonized = cythoinzed

    def fit(self, X, Y):
        if self.cythonized is True:
            if (Y.shape[1] == 1):
                self.rank = np.linalg.matrix_rank(X)
                self.W, self.B, self.P = pls_cy.PLS_cython_1d(X, Y, self.rank)
                self.dim = 1

            else:
                self.rank = np.linalg.matrix_rank(X)
                self.W, self.B, self.P, self.Q = pls_cy.PLS_cython(X, Y, self.rank, 1000000, np.random.randint(0, Y.shape[1]))
                self.dim = Y.shape[1]
        else:
            if(Y.shape[1] == 1):
                self.W, self.B, self.P = pls_1d.PLS1(X, Y)
                self.dim = 1
                self.rank = np.linalg.matrix_rank(X)
            else:

                self.W, self.B, self.P, self.Q = pls_nd.PLS(X, Y)
                self.dim = Y.shape[1]
                self.rank = np.linalg.matrix_rank(X)




    def predict(self, X):
        if self.dim is 1 :
            r = np.linalg.matrix_rank(X)
            Q = np.ones((1, r))
            E_h = X
            y_pred = np.zeros((X.shape[0], 1))
            for i in range(r):
                t_hat = E_h @ self.W[i]
                E_h = E_h - t_hat @ self.P[i].T
                y_pred = y_pred + self.B[i] * t_hat
            return y_pred
        else:

            E = X
            T_hat = np.zeros(X.shape)
            Y_pred = np.zeros((X.shape[0], self.dim))

            for i in range(self.rank):
                T_hat[:, i] = E @ self.W[i]

                E = E - T_hat[:, i][:, None] @ self.P[i][:, None].T
                inter = T_hat[:, i][:, None] @ self.Q[i][None, :]
                Y_pred += self.B[i] * inter
            return Y_pred



