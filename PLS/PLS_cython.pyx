import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport sqrt
from cython.parallel import prange, parallel
import cython
cimport cython

ctypedef np.double_t DTYPE_t
ctypedef np.int64_t TTYPE_t
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef fast_dot(float[:, :] A, float[:, :] B):
    '''prepare data'''
    cdef int A_row = A.shape[0]
    cdef int A_col = A.shape[1]
    cdef int B_row = B.shape[0]
    cdef int B_col = B.shape[1]
    cdef float[:, :] mat_1 = A
    cdef float[:,:] mat_2 = B
    cdef float[:, :] result = np.zeros([A.shape[0],B.shape[1]],dtype=np.float32)
    '''begin dot'''
    cdef int i=0, j=0, k=0
    for i in range(A_row):
        for j in range(B_col):
            result[i,j] = 0.0
            for k in range(A_col):
                result[i,j] += mat_1[i,k] * mat_2[k,j]
#     print(np.multiply(result, 1))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef fast_dot_n1(float[:, :] A, float[:, :] B):
    '''prepare data'''
    cdef int A_row = A.shape[0]
    cdef int A_col = A.shape[1]
    cdef int B_row = B.shape[0]
    cdef int B_col = 1
    cdef float[:, :] mat_1 = A
    cdef float[:,:] mat_2 = B
    cdef float[:] result = np.zeros(A.shape[0],dtype=np.float32)
    '''begin dot'''
    cdef int i=0, j=0, k=0
    for i in range(A_row):
        for j in range(1):
            result[i] = 0.0
            for k in range(A_col):
                result[i] += mat_1[i,k] * mat_2[k,0]
#     print(np.multiply(result, 1))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef fast_dot_11(float[:, :] A, float[:, :] B):
    '''prepare data'''
    cdef int A_col = A.shape[1]
    cdef int B_row = B.shape[0]
    cdef float[:,:] mat_1 = A
    cdef float[:,:] mat_2 = B
    cdef float result = 0.0
    '''begin dot'''
    cdef int i=0, j=0, k=0
    for i in range(A_col):
        result += mat_1[0,i] * mat_2[i,0]
#     print(np.multiply(result, 1))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef scalar_multiply(float a, float[:, :] b):
    cdef float[:, :] mat = b
    cdef int blen = b.shape[0]
    cdef int bwid = b.shape[1]
    cdef int i,j
    for i in range(blen):
        for j in range(bwid):
            mat[i, j] = a*b[i, j]
    return mat

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef scalar_division(float[:] vec, float sca):
    cdef float[:] mat = vec
    cdef int blen = vec.shape[0]
    cdef int i
    for i in range(blen):
        mat[i] = vec[i]/sca
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef scalar_division_1d2d(float[:,:] vec, float sca):
#     ans = np.zeros([vec.shape[0], 1]).astype(np.float32)
    cdef float[:, :] mat = np.zeros([vec.shape[0], 1],dtype=np.float32)
    cdef int blen = vec.shape[0]
    cdef int i
    for i in range(blen):
        mat[i,0] = vec[i,0]/sca
    return mat

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef minus_2d(float[:,:] A, float[:,:] B):
    cdef float[:, :] mat = np.zeros([A.shape[0], A.shape[1]],dtype=np.float32)
    cdef int i,j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            mat[i,j] = A[i,j] - B[i,j]
    return mat

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef dot_1d(float[:,:] v1, float[:,:] v2):
    cdef float result = 0.0
    cdef int i = 0
    cdef int length = v1.shape[0]
    cdef double el1 = 0
    cdef double el2 = 0
    for i in range(length):
        el1 = v1[i,0]
        el2 = v2[0,i]
        result += el1*el2
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef float norm_1d(float[:,:] v1):
    cdef float result = 0.0
    cdef int i = 0
    cdef int length = v1.shape[0]
    for i in range(length):
        result += v1[i,0]*v1[i,0]
    result = sqrt(result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef PLS_cython(float[:,:] X, float[:,:] Y,int r,int max_iter,int rand_col):
#     X = X.astype(np.float32)
#     Y = Y.astype(np.float32)
    cdef float[:, :] E_h = X
#     cdef float[:, :] F_h = Y.reshape([Y.shape[0], 1])
    cdef float[:, :] F_h = Y
    W = {}
    B = {}
    P = {}
    Q = {}
    cdef float[:,:] u = np.zeros([Y.shape[0],1],dtype=np.float32)
    cdef float[:,:] w = np.zeros([X.shape[1],1],dtype=np.float32)
    cdef float[:, :] t = np.zeros([X.shape[0], 1],dtype=np.float32)
    cdef float[:, :] t_ = np.zeros([X.shape[0], 1],dtype=np.float32)
    cdef float[:, :] p = np.zeros([X.shape[1], 1],dtype=np.float32)
    cdef float[:, :] q = np.zeros([Y.shape[1], 1],dtype=np.float32)
    cdef float[:, :] q_ = np.zeros([Y.shape[1], 1],dtype=np.float32)
    cdef float b = 0.0
    cdef int i,j,k
    cdef int yshape
    cdef float pnorm
    for i in range(r):

        yshape = Y.shape[0]
        for k in range(yshape):
            u[k,0] = Y[k,rand_col]
        for j in range(max_iter):
            w = np.dot(X.T, u)/ dot_1d(u.T ,u)
            #step 3
            w = scalar_division_1d2d(w,norm_1d(w))

            t_ = t
            #step 4

            t = np.dot(X, w)/dot_1d(w.T, w)
            q = np.dot(Y.T, t)/dot_1d(t.T, t)
            q_ = scalar_division_1d2d(q,norm_1d(q))
            u = np.dot(Y, q)/dot_1d(q.T, q)
            if np.allclose(t_, t, rtol = 1e-6, atol = 1e-6):
                break

        p = np.dot(X.T, t)/dot_1d(t.T, t)
        #step 10
        pnorm = norm_1d(p)
        p = scalar_division_1d2d(p, pnorm)


        #step 11
        t = scalar_multiply(pnorm,t)
        #step 12
        w = scalar_multiply(pnorm,w)
        #step 13
        b = np.dot(u.T, t)/dot_1d(t.T, t)

        W[i] = w
        B[i] = b
        P[i] = p
        Q[i] = q

        E_h = minus_2d(E_h,np.dot(t,p.T))
        F_h = minus_2d(F_h,np.dot(scalar_multiply(b,t),q.T))

        X = E_h
        Y = F_h

    return W,B,P,Q


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef predict_cython(float[:,:] X, float[:,:] Y, float[:,:] X_test,int r):
    W,B,P,Q = PLS_cython(X, Y, r, 10000000, np.random.randint(0, Y.shape[1]))
    cdef float[:,:] E = X
    cdef float[:,:] T_hat = np.zeros([X_test.shape[0], 1]).astype(np.float32)
    cdef float[:,:] Y_pred = np.zeros([X_test.shape[0], Y.shape[1]]).astype(np.float32)
    cdef int i,j,k
    for i in range(r):
        T_hat = np.dot(E,W[i])
        E = E - np.dot(T_hat, P[i].T)
        inter = np.dot(T_hat ,Q[i].T)
        for k in range(Y_pred.shape[0]):
            for j in range(Y_pred.shape[1]):
                Y_pred[k,j] += B[i] * inter[k,j]
    return Y_pred


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef PLS_cython_1d(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y, int r):
    cdef np.ndarray[DTYPE_t, ndim=2] E_h = X.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] F_h = Y.copy()
    W = {}
    B = {}
    P = {}
    cdef np.ndarray[DTYPE_t, ndim=2] u = Y.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] w = np.zeros([X.shape[1],1])
    cdef np.ndarray[DTYPE_t, ndim=2] t = np.zeros([X.shape[0], 1])
    cdef np.ndarray[DTYPE_t, ndim=2] p = np.zeros([X.shape[1], 1])
    cdef DTYPE_t b = 0.0
    cdef int i


    for i in range(r):
        u = Y
        w = scalar_division_1d2d(np.dot(X.T, u), dot_1d(u,u.T))

        #step 3
        w = w/norm_1d(w)

        #step 4

        t = scalar_division_1d2d(np.dot(X, w),dot_1d(w, w.T))
        #step5-8 omitted
        #step 9
        p = scalar_division_1d2d(np.dot(X.T, t),dot_1d(t,t.T))
        p_norm = norm_1d(p)
        #step 10
        p = p/p_norm
        #step 11
        t = t* p_norm
        #step 12
        w = w * p_norm
        #step 13
        b = np.dot(u.T, t)/dot_1d(t,t.T)
#         print(b.shape)
        # Calculation of the residuals

        E_h = minus_2d(E_h,np.dot(t,p.T))
        F_h = minus_2d(F_h,scalar_multiply(b,t))
#         print(F_h.shape)
        #Replace X and Y
        X = E_h
        Y = F_h
        #update W and B
        W[i] = w
        B[i] = b
        P[i] = p
    return W,B,P

cpdef predict_cython_1d(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y,np.ndarray[DTYPE_t, ndim=2] X_test,int r):
    W,B,P = PLS_cython_1d(X,Y, r)
    cdef np.ndarray[DTYPE_t, ndim=2] E_h = X_test.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] y_pred = np.zeros((X_test.shape[0],1))
    cdef np.ndarray[DTYPE_t, ndim=2] t_hat = np.zeros((X_test.shape[0],1))
    cdef int i,j
    for i in range(r):
        t_hat = np.dot(E_h, W[i])
        E_h = E_h - np.dot(t_hat, P[i].T)
        for j in range(y_pred.shape[0]):
            y_pred[j,0] = y_pred[j,0] + B[i] * t_hat[j,0]
    return y_pred[:,0]


cpdef predict_1d(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y,np.ndarray[DTYPE_t, ndim=2] X_test,int r):
    W,B,P = PLS_cython_1d(X,Y, r)
    cdef np.ndarray[DTYPE_t, ndim=2] E_h = X_test.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] y_pred = np.zeros((X_test.shape[0],1))
    cdef np.ndarray[DTYPE_t, ndim=2] t_hat = np.zeros((X_test.shape[0],1))
    cdef int i,j
    for i in range(r):
        t_hat = np.dot(E_h, W[i])
        E_h = E_h - np.dot(t_hat, P[i].T)
        for j in range(y_pred.shape[0]):
            y_pred[j,0] = y_pred[j,0] + B[i] * t_hat[j,0]
    return y_pred[:,0]