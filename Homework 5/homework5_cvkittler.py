import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

def phiPoly3 (x):
    radon = x[:, 0]
    asbestos = x[:, 1]

    d0 = np.ones(asbestos.shape)
    d9 = np.sqrt(3) * asbestos
    d6 = np.sqrt(3) * radon
    d5 = np.sqrt(6) * radon * asbestos.T
    d8 = np.sqrt(3) * asbestos ** 2
    d3 = np.sqrt(3) * radon ** 2
    d2 = np.sqrt(3) * radon ** 2 * asbestos.T
    d4 = np.sqrt(3) * radon.T * asbestos ** 2
    d7 = asbestos ** 3
    d1 = radon ** 3
    phi = np.array([d0,
                    d9,
                    d6,
                    d5,
                    d8,
                    d3,
                    d2,
                    d4,
                    d7,
                    d1])
    return phi

def kerPoly3 (x, xprime):
    X = np.array(x)
    Xprime = np.array(xprime)
    return (np.dot(X, Xprime.T) + 1) ** 3

def showPredictions (title, svm, X, increment_y = 2, increment_x = .1):  # feel free to add other parameters if desired
    positiveExample = np.zeros([2,2])
    negativeExample = np.zeros([2,2])

    for x_cord in [float(i) * increment_x for i in range(0,int(10/increment_x))]:
        for y_cord in [float(j) * increment_y for j in range(int(50/increment_y),int(200/increment_y))]:
            if X.shape[1] != 2:
                input = phiPoly3(np.array([[x_cord,y_cord],[x_cord,y_cord]]))
                prediction = svm.predict([input[:,1]])
            else:
                prediction = svm.predict([[x_cord,y_cord]])
            if prediction > 0:
                positiveExample = np.append(positiveExample, np.array([[x_cord,y_cord]]), axis=0)
            else:
                negativeExample = np.append(negativeExample, np.array([[x_cord,y_cord]]), axis=0)
    positiveExample = positiveExample[2:,:]
    negativeExample = negativeExample[2:, :]
    plt.scatter(positiveExample[:,0], positiveExample[:, 1])
    plt.scatter(negativeExample[:, 0], negativeExample[:, 1])
    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend([ "Lung disease", "No lung disease" ])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Load training data
    d = np.load("lung_toy.npy")
    X = d[:,0:2]  # features
    y = d[:,2]  # labels

    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend(["Lung disease", "No lung disease"])
    plt.title("Data")
    plt.show()

    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    showPredictions("Linear", svmLinear, X)

    # (b) Poly-3 using explicit transformation phiPoly3
    xTrans = phiPoly3(X)
    svmLinearTrans = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinearTrans.fit(xTrans.T, y)
    showPredictions("Explicit Transformation", svmLinearTrans, xTrans)

    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    svmLinearKern = sklearn.svm.SVC(kernel=kerPoly3, C=0.01)
    svmLinearKern.fit(X, y)
    showPredictions("Kernel Matrix", svmLinearKern, X)

    # (d) Poly-3 using sklearn's built-in polynomial kernel
    svmPoly = sklearn.svm.SVC(kernel='poly', C=0.01, gamma=1, coef0 = 1, degree = 3)
    svmPoly.fit(X, y)
    showPredictions("sklearn Built in Poly", svmPoly, X)

    # (e) RBF using sklearn's built-in polynomial kernel
    svmRBF1 = sklearn.svm.SVC(kernel='rbf', C=1, gamma = 0.1)
    svmRBF1.fit(X, y)
    showPredictions("sklearn Built in RBF γ = 0.1", svmRBF1, X)

    svmRBF2 = sklearn.svm.SVC(kernel='rbf', C=1, gamma = 0.03)
    svmRBF2.fit(X, y)
    showPredictions("sklearn Built in RBF γ = 0.03", svmRBF2, X)

    print("Done!")