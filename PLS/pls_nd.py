import numpy as np
import numba
from numba import jit
import pandas as pd

def PLS(X, Y):
    t_ = 0
    t = 0
    W = {}
    B = {}
    P = {}
    Q = {}
    E = X
    F = Y

    rank_x = np.linalg.matrix_rank(X)
    max_iter = 1000000
    rand_col = np.random.randint(0, F.shape[1])

    for i in range(rank_x):
        u = F[:, rand_col]
        for _ in range(max_iter):
            w = np.dot(u.T, X) / np.dot(u.T, u)
            w = w / np.linalg.norm(w)
            t_ = t
            t = np.dot(E, w) / np.dot(w.T, w)
            q = np.dot(t.T, F) / np.dot(t.T, t)
            q_ = q / np.linalg.norm(q)
            u = np.dot(F, q.T) / np.dot(q.T, q)
            if (np.allclose(t_, t, rtol=1e-3, atol=1e-3)):
                break

        p = np.dot(t.T, E) / np.dot(t, t.T)
        p = p / np.linalg.norm(p)
        t = t * np.linalg.norm(p)
        w = w * np.linalg.norm(p)

        b = np.dot(u.T, u) / np.dot(t.T, t)
        W[i] = w
        B[i] = b
        P[i] = p
        Q[i] = q

        t_ = t.reshape([-1, 1])
        p_ = p.reshape([-1, 1])
        q_ = q.reshape([-1, 1])
        E = E - t_ @ p_.T
        F = F - b * t_ @ q_.T

    return W, B, P, Q


@jit
def all_close(t, t_old, tol=1e-5):
    return np.sum(np.abs(t - t_old)) < tol


@jit(nopython=True)
def PLS_jit(X, Y):
    t_old = X[:, 0]
    t = X[:, 0]
    rank_x = np.linalg.matrix_rank(X)
    W = np.zeros((X.shape[1], rank_x), np.float64)
    Q = np.zeros((Y.shape[1], rank_x), np.float64)
    P = np.zeros((X.shape[1], rank_x), np.float64)
    B = np.zeros(rank_x, np.float64)
    E = X.astype(np.float64)
    F = Y.astype(np.float64)

    y_shape = Y.shape[1]
    max_iter = 10000000
    rand_col = np.random.randint(0, F.shape[1])

    for i in range(rank_x):
        u = F[:, rand_col]
        for _ in range(max_iter):
            w = u.T @ E / (u.T @ u)
            w = w / np.linalg.norm(w)
            t_old = t
            t = E @ w / (w.T @ w)
            q = t.T @ F / (t.T @ t)
            q_ = q / np.linalg.norm(q)
            u = F @ q.T / (q.T @ q)
            if (all_close(t_old, t)):
                break

        p = (t.T @ E) / (t @ t.T)
        p = p / np.linalg.norm(p)
        t = t * np.linalg.norm(p)
        w = w * np.linalg.norm(p)

        b = (u.T @ u) / (t.T @ t)
        W[i] = w
        B[i] = b
        P[i] = p
        Q[i] = q

        t = np.copy(t.T[0])
        p = np.copy(p.T[0])
        q = np.copy(q.T[0])

        t = t.reshape((-1, 1))
        p = p.reshape((-1, 1))
        q = q.reshape((-1, 1))

        E = E - t @ p.T
        F = F - b * t @ q.T

    return W, B, P, Q


np.random.seed(9856)
x1 = np.random.normal(1, .2, 100)
x2 = np.random.normal(5, .4, 100)
x3 = np.random.normal(12, .8, 100)


def generate_sim(x1, x2, x3):
    sim_data = {'x1': x1,
                'x2': x2,
                'x3': x3,
                'x4': 5 * x1,
                'x5': 2 * x2,
                'x6': 4 * x3,
                'x7': 6 * x1,
                'x8': 5 * x2,
                'x9': 4 * x3,
                'x10': 2 * x1,
                'y0': 3 * x2 + 3 * x3,
                'y1': 6 * x1 + 3 * x3,
                'y2': 7 * x2 + 2 * x1}

    # convert data to csv file
    data = pd.DataFrame(sim_data)

    sim_predictors = data.drop(['y0', 'y1', 'y2'], axis=1).columns.tolist()
    sim_values = ['y0', 'y1', 'y2']

    pred = data[sim_predictors].values
    val = data[sim_values].values

    return pred, val


pred, val = generate_sim(x1, x2, x3)

test_x1 = np.random.normal(1, .2, 100)
test_x2 = np.random.normal(5, .4, 100)
test_x3 = np.random.normal(12, .8, 100)

pred_test, pred_val = generate_sim(test_x1, test_x2, test_x3)

pls_jit_speed = PLS_jit(pred, val)



