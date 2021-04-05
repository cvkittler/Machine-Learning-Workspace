import numpy as np


def problem1(A, B):
    return A + B


def problem2(A, B, C):
    return np.dot(A, B) - C


def problem3(A, B, C):
    return A * B + np.transpose(C)


def problem4(x, S, y):
    xT = np.transpose(x)
    xTS = np.dot(xT, S)
    return np.dot(xTS, y)


def problem5(A):
    return np.zeros(A.shape, type(A.ndim))


def problem6(A):
    return np.ones(A.shape[0], type(A.ndim))


def problem7(A, alpha):
    I = np.eye(A.shape[0])
    alphaI = alpha * I
    return A + alphaI


def problem8(A, i, j):
    return A[i, j]


def problem9(A, i):
    return np.sum(A[i, ])


def problem10(A, c, d):
    a1 = A[np.nonzero(A >= c)]
    a2 = a1[np.nonzero(a1 <= d)]
    return np.mean(a2)


def problem11(A, k):
    eigenVectors = np.linalg.eig(A)
    ind = np.unravel_index(np.argsort(eigenVectors[0]), eigenVectors[0].shape)
    vectors = eigenVectors[1]
    sortedVectors = vectors[ind]
    return sortedVectors[0:k]


def problem12(A, x):
    return np.linalg.solve(A, x)


def problem13(A, x):
    return np.linalg.solve(x, A)
