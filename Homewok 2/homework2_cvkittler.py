import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# Data from
########################################################################################################################
# https://s3.amazonaws.com/jrwprojects/age_regression_Xtr.npy
# https://s3.amazonaws.com/jrwprojects/age_regression_ytr.npy
# https://s3.amazonaws.com/jrwprojects/age_regression_Xte.npy
# https://s3.amazonaws.com/jrwprojects/age_regression_yte.npy

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    pass

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    shape = faces.shape
    reshaped = faces.reshape(-1, shape[1]*shape[1])
    returnMe = np.append(reshaped, np.ones((shape[0], 1)), axis=1)
    return returnMe.T

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    yHat = np.dot(Xtilde.T, w)
    return ((y - yHat) **2).mean(axis=0)


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    xxT = np.dot(Xtilde, Xtilde.T)
    Xy = np.dot(Xtilde, y)
    w = np.linalg.solve(xxT, Xy)
    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde,y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, alpha=ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 0.0003  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    n = Xtilde.shape[0]
    w = 0.001 * np.random.randn(n)
    w = np.array(w, dtype=np.double)
    y = np.array(y, dtype=np.double)

    for i in range(T):
        xTw = np.dot(Xtilde.T, w)
        gradient = np.dot(Xtilde, xTw  - y)/n
        gradient = gradient + (alpha/(2*n)) * np.dot(w.T, w)
        w = w - (gradient * EPSILON)
        if i % 100 == 0:
            print("T: " + str(i) + "\tWeights 1-6" + str(w[:6]))
    return w

def showWeightsAsImage(w):
    im = w[:-1].reshape([48, 48])
    fig, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    plt.show()

def findWorstPredictions(w, X, y, amount_worst=5):
    yHat = np.dot(X.T, w)
    errorDiff = y - yHat
    mse = errorDiff**2
    indexOrder = np.argsort(mse)
    errSorted = np.take_along_axis(mse, indexOrder, axis=0)
    imageIndex = np.zeros(X.shape,dtype=np.int32)
    imageIndex[:,:] = indexOrder
    imgSorted = np.take_along_axis(X, imageIndex, axis=1)
    for i in range(amount_worst):
        print("MSE: " + str(errSorted[-i-1] * errSorted[-i-1]))
        showWeightsAsImage(imgSorted[:,-i-1])

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")
    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    print("Method 1 Training:")
    print(fMSE(w1,Xtilde_tr, ytr))
    print("Method 1 Testing:")
    print(fMSE(w1, Xtilde_te, yte))
    showWeightsAsImage(w1)
    print("\nMethod 2 Training:")
    print(fMSE(w2, Xtilde_tr, ytr))
    print("Method 2 Testing:")
    print(fMSE(w2, Xtilde_te, yte))
    showWeightsAsImage(w2)
    print("\nMethod 3 Training:")
    print(fMSE(w3, Xtilde_tr, ytr))
    print("Method 3 Testing:")
    print(fMSE(w3, Xtilde_te, yte))
    showWeightsAsImage(w3)

    findWorstPredictions(w3, Xtilde_te, yte)