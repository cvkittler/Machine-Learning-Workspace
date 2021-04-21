import numpy as np
import matplotlib.pyplot as plt

# Data Location
# • https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy
# • https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy
# • https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy
# • https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None, numEpochs = 1):
    if batchSize is None:
        batchSize = trainingLabels.size

    # start with a random weights
    w = 0.001 * np.random.randn(trainingImages.shape[1], 10)

    numBatches = int(trainingLabels.size / batchSize) - 1
    dataLoss = np.zeros([20, trainingImages.shape[1], 10])

    for _ in range(numEpochs):
        # reorder images
        newOrder = np.random.shuffle(np.arange(trainingLabels.size))
        trainingLabelsNewOrder = trainingLabels[newOrder]
        trainingImagesNewOrder = trainingImages[newOrder]
        # Batches
        for e in range(numBatches):
            trainingImagesBatch = trainingImages[e*batchSize:(e+1)*batchSize]
            trainingLabelsBatch = trainingLabels[e*batchSize:(e+1)*batchSize]
            w = gradientDescentSoftMax(trainingImagesBatch, trainingLabelsBatch, w, EPSILON=epsilon)
            dataLoss[:-1] = dataLoss[1:]
            dataLoss[-1] = w
    return w, dataLoss

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def append1s (images):
    shape = images.shape
    images = np.append(images, np.ones((shape[0], 1)), axis=1)
    return images

# Helper method for Gradient Descent
def gradientDescentSoftMax (Images, Labels, Weights, EPSILON = 0.0003):
    # z = X.T (dot) W
    #np.exp
    # np.sum then element wise division
    # gradiant = x(yHat - y)

    #transform the labels into a usable form (Y)
    n = Labels.size
    y = np.zeros([n, 10])
    y[np.arange(n), Labels] = 1

    z = np.dot(Images, Weights)
    expZ = np.exp(z)
    sigmaExpZ = np.sum(expZ, axis=1)
    yHat = expZ/np.vstack(sigmaExpZ)
    gradient = np.dot(Images.T, (yHat - y)) / n
    return Weights - (EPSILON * gradient)

def PC(w, images, y):
    z = np.dot(images, w)
    expZ = np.exp(z)
    sigmaExpZ = np.sum(expZ, axis=1)
    yHat = expZ / np.vstack(sigmaExpZ)
    guess = np.argmax(yHat, axis=1)
    correct = np.equal(guess, y)

    return np.sum(correct)/y.size

def CE(w, images, Labels):
    # transform the labels into a usable form (Y)
    n = Labels.size
    y = np.zeros([n, 10])
    y[np.arange(n), Labels] = 1

    z = np.dot(images, w)
    expZ = np.exp(z)
    sigmaExpZ = np.sum(expZ, axis=1)
    yHat = expZ / np.vstack(sigmaExpZ)
    logYHat = np.log(yHat)

    return np.sum(np.sum(y * logYHat, axis=1)) * (-1/n)

def showWeightAsImage(w):
    for classification in range(10):
        im = w[:-1,classification].reshape([28, 28])
        fig, ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        plt.show()

def graphData(data, images, Labels):
    plotData = np.array(10)
    for wSet in range(20):
        plotData = np.append(plotData, CE(data[wSet,:,:], images, Labels))
    plt.plot(plotData[1:])
    plt.axis([0,20,0,1])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy")
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy")
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Normalize Pixels
    trainingImages = trainingImages / 255
    testingImages = testingImages / 255

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = append1s(trainingImages)
    testingImages = append1s(testingImages)

    W, ceLossData = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, numEpochs = 5)
    print(PC(W,testingImages,testingLabels))
    print(CE(W, testingImages, testingLabels))
    # Visualize the vectors

    showWeightAsImage(W)
    # graphData(ceLossData,trainingImages,trainingLabels)
    pass