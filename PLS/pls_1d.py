import numpy as np

def PLS1(X, Y):
    r = np.linalg.matrix_rank(X)
    E_h = X
    F_h = Y.reshape(-1, 1)
    W = {}
    B = {}
    P = {}
    for i in range(r):
        u = Y
        w = (X.T @ u) / (u.T @ u)
        w = w / np.linalg.norm(w)
        p = (X.T @ t) / (t.T @ t)
        p = p / np.linalg.norm(p)
        t = t * np.linalg.norm(p)
        w = w * np.linalg.norm(p)
        b = (u.T @ t) / (t.T @ t)
        t = t.reshape((-1, 1))
        p = p.reshape((-1, 1))
        E_h = E_h - t @ p.T
        F_h = F_h - b * t
        X = E_h
        Y = F_h
        W[i] = w
        B[i] = b
        P[i] = p
    return W, B, P
