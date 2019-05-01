import numba
from numba import jit, vectorize, float64, int64
import numpy as np

def nipals(X):
    for i in range(X.shape[1]):
        t_h = X[:, i]
        p_h = (t_h.T@X)/(t_h.T@t_h)
        p_h = p_h/np.norm(p_h)
        t_new = X@p_h/(p_h.T@p_h)
        if(np.allclose(t_h, t_new)):
            return t_h, p_h

@jit
def update1(X,u,t,t_,E,F):
    w = np.dot(u.T, X)/np.dot(u.T, u)
    w = w/np.linalg.norm(w)
    t_ = t
    t = np.dot(E, w)/np.dot(w.T,w)
    q = np.dot(t.T, F)/np.dot(t.T,t)
    q_ = q/np.linalg.norm(q)
    u = np.dot(F,q.T)/np.dot(q.T,q)
    return w,t,t_,q,u


@jit
def update2(t, w, u, q, i, E, F):
    p = np.dot(t.T, E) / np.dot(t, t.T)
    p = p / np.linalg.norm(p)
    t = t * np.linalg.norm(p)
    w = w * np.linalg.norm(p)

    b = np.dot(u.T, u) / np.dot(t.T, t)
    #     W[i] = w
    #     B[i] = b
    #     P[i] = p
    #     Q[i] = q

    t_ = t.reshape([-1, 1])
    p_ = p.reshape([-1, 1])
    q_ = q.reshape([-1, 1])
    E = E - t_ @ p_.T
    F = F - b * t_ @ q_.T
    return E, F, w, b, p


def PLS(X, Y):
    t_ = 0
    t = 0
    W = {}
    B = {}
    P = {}
    Q = {}
    E = X
    F = Y

    y_shape = Y.shape[1]
    rank_x = np.linalg.matrix_rank(X)
    max_iter = 10000000
    rand_col = np.random.randint(0, F.shape[1])

    for i in range(rank_x):
        u = F[:, rand_col]
        for _ in range(max_iter):

            w, t, t_, q, u = update1(X, u, t, t_, E, F)
            if (np.allclose(t_, t, rtol=1e-6, atol=1e-6)):
                break


        E, F, w, b, p = update2(t, w, u, q, i, E, F)
        W[i] = w
        B[i] = b
        P[i] = p
        Q[i] = q
    return W, B, P, Q