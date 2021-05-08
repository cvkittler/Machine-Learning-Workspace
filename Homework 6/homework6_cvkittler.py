import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    constantA = NUM_HIDDEN * NUM_INPUT
    W1 = w[:constantA]
    constantB = NUM_HIDDEN + constantA
    b1 = w[constantA:constantB]
    constantC = constantB + NUM_OUTPUT * NUM_HIDDEN
    W2 = w[constantB:constantC]
    b2 = w[constantC:]
    W1 = np.reshape(W1,[NUM_HIDDEN,NUM_INPUT])
    W2 = np.reshape(W2,[NUM_OUTPUT,NUM_HIDDEN])
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.append(W1,np.append(b1,np.append(W2,b2)))

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    return images, labels

def PC(X, Y, W):
    result = np.argmax(calcYHat(X.T, W), axis=0) == np.argmax(Y, axis=1)
    return np.sum(result) / result.shape[0]

def calcYHat(_X, _w):
    _W1, _b1, _W2, _b2 = unpack(_w)

    z1 = (np.dot(_W1 , _X).T + _b1).T
    h1 = np.where(z1 > 0, z1, 0)
    z2 = (np.dot(_W2 , h1).T + _b2).T
    yHat = softmax(z2)

    return yHat

#calc the softmax
def softmax(x):
    x.astype(np.longdouble)
    exp = np.exp(x)
    exp_sum = np.sum(exp, axis=0)
    return (exp / exp_sum)

# Computes cross entropy for all values in X, Y
def fCE(_X, _Y, _w):
    _X = _X.T
    yHat = calcYHat(_X, _w)
    logYHat = np.log(yHat)
    sumLogYHat = np.sum(_Y.T * logYHat)

    cost = (-1/_X.shape[1]) * sumLogYHat
    return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(_X, y, _w):
    X = _X.T
    W1, b1, W2, b2 = unpack(_w)

    z1 = (np.dot(W1 , X).T + b1).T
    h1 = np.where(z1 > 0, z1, 0)
    z2 = (np.dot(W2 , h1).T + b2).T
    yhat = softmax(z2)
    yhat_y = yhat - y.T
    gT = np.dot(yhat_y.T , W2) * np.where(z1.T >= 0, 1.0, 0.0)
    g = gT.T

    grad_w2 = np.dot(yhat_y , h1.T)
    grad_b2 = np.mean(yhat_y, axis=1)
    grad_w1 = np.dot(g , X.T)
    grad_b1 = np.mean(g, axis=1)

    grad = pack(grad_w1, grad_b1, grad_w2, grad_b2)
    return grad

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX_, trainY_, testX, testY, _w, L1_coeff = 0.0, L2_coeff = 0.0001, batchSize = 8, numEpochs = 30, learning_rate = 0.01):

    numImages = trainX_.shape[0]
    if batchSize is None:
        batchSize = numImages
    # change the order of the examples
    newOrder = np.arange(numImages)
    np.random.shuffle(newOrder)
    X = trainX_[newOrder, :]
    Y = trainY_[newOrder, :]

    w_past = np.zeros([numEpochs, _w.size])
    for EPOCH in range(numEpochs):
        for batch_start in range(0, numImages, batchSize):
            if(batch_start + batchSize > numImages):
                batchSize = numImages - batch_start
            xBatch = X[batch_start:batch_start + batchSize,:]
            yBatch = Y[batch_start:batch_start + batchSize,:]

            gradients = gradCE(xBatch,yBatch, _w)
            w1_delta, b1_delta, w2_delta, b2_delta = unpack(gradients)
            _w1, _b1, _w2, _b2 = unpack(_w)


            _w1 -= learning_rate * w1_delta + L1_coeff * np.sign(_w1) + L2_coeff * _w1
            _w2 -= learning_rate * w2_delta + L1_coeff * np.sign(_w2) + L2_coeff * _w2
            _b1 -= learning_rate * b1_delta + L1_coeff * np.sign(_b1) + L2_coeff * _b1
            _b2 -= learning_rate * b2_delta + L1_coeff * np.sign(_b2) + L2_coeff * _b2

            _w = pack (_w1, _b1, _w2, _b2)

        w_past[EPOCH] = _w
        # print("Epoch:", EPOCH, "Test Score:", PC(testX, testY, _w))
    return _w, w_past

def graphData(old_w, X, Y, PERAMS_STRING):
    numRecorded = old_w.shape[0]
    accuracy = np.zeros(numRecorded)
    cross_entropy = np.zeros(numRecorded)
    for cur in range(numRecorded):
        accuracy[cur] = PC(X,Y,old_w[cur])
        cross_entropy[cur] = fCE(X,Y,old_w[cur])
    plt.plot(cross_entropy)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    title = 'Loss vs Epoch' + PERAMS_STRING
    plt.title(title)
    fileName = 'plots/' + str(title) + '.jpeg'
    plt.savefig(fileName)
    plt.show()

    plt.plot(accuracy)
    plt.ylabel('Percent Correct')
    plt.xlabel('Epoch')
    title = 'Percent Correct vs Epoch' + PERAMS_STRING
    plt.title(title)
    fileName = 'plots/' + str(title) + '.jpeg'
    plt.savefig(fileName)
    plt.show()

def findBestHyperparameters(X, Y, W):
    #unused
    L1_coeff = np.array([0, 0.00001, 0.0001, 0.001, 0.01])
    L2_coeff = np.array([0, 0.00001, 0.0001, 0.001, 0.01])
    unitsHiddenLayer = np.array([30,40,50])

    #used
    batchSize = np.array([16, 32, 64, 128])
    bestBatchSize = batchSize[0]
    numEpochs = np.array([40, 50, 60])
    bestNumEpochs = numEpochs[0]
    learning_rate = np.array([0.0001, 0.001, 0.005, 0.01])
    best_learning_rate = learning_rate[0]

    numImages = trainX.shape[0]

    bestPC = 0
    for batchSize_ in batchSize:
        for numEpochs_ in numEpochs:
            for learning_rate_ in learning_rate:
                newOrder = np.arange(numImages)
                np.random.shuffle(newOrder)
                X = X[newOrder, :]
                Y = Y[newOrder, :]

                X_test = X[:int(numImages * 0.2),:]
                X_train = X[int(numImages * 0.2):,:]
                Y_test = Y[:int(numImages * 0.2),:]
                Y_train = Y[int(numImages * 0.2):,:]

                w_found,w_past = train(X_train, Y_train, testX, testY, W, batchSize=batchSize_,numEpochs=numEpochs_,learning_rate=learning_rate_)
                titel_add_on = "BatchSize" + str(batchSize_) + "Epochs" + str(numEpochs_) + "LearnRate" + str(learning_rate_)
                # graphData(w_past, X_test, Y_test, titel_add_on)

                if PC(X_test, Y_test, w_found) > bestPC and fCE(X_test, Y_test, w_found) == fCE(X_test, Y_test, w_found):
                    print(titel_add_on)
                    bestBatchSize = batchSize_
                    bestNumEpochs = numEpochs_
                    best_learning_rate = learning_rate_
                    bestW = w_found
                    bestPC = PC(X_test, Y_test, w_found)
    print("Best Hyper Parameters:", " Batch Size: " + str(bestBatchSize) + " Number of Epochs: " + str(bestNumEpochs) + " Learning Rate: " + str(best_learning_rate))
    return bestW, titel_add_on
if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    trainX = trainX / 255
    testX = testX / 255

    n = trainY.size
    Y = np.zeros([n, 10])
    Y[np.arange(n), trainY] = 1
    trainY = Y
    n = testY.size
    Y = np.zeros([n, 10])
    Y[np.arange(n), testY] = 1
    testY = Y

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    # idxs = np.random.permutation(trainX.shape[0])[0: 1]
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
    #                                 w))

    # Train the network using SGD.
    w,w_past = train(trainX, trainY, testX, testY, w, batchSize=128, numEpochs=40, learning_rate=0.001)
    graphData(w_past, trainX, trainY, ' Default')


    # UNCOMMENT TO DO HYPER PERAMETER SEARCH
    # w_found, titel = findBestHyperparameters(trainX, trainY, w)
    # print(PC(testX,testY, w_found))
    # print(fCE(testX, testY, w_found))
    # graphData(w_found, testX, trainY, titel)