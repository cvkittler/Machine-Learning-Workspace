import numpy as np
import pandas

# Helper method for Gradient Descent
def gradientDescentSoftMax (Images, Labels, Weights, EPSILON = 0.0003, numFeatures=2):
    # z = X.T (dot) W
    #np.exp
    # np.sum then element wise division
    # gradiant = x(yHat - y)

    #transform the labels into a usable form (Y)
    n = Labels.size
    y = np.zeros([n, numFeatures])
    y[np.arange(n), Labels] = 1

    z = np.dot(Images, Weights)
    expZ = np.exp(z)
    sigmaExpZ = np.sum(expZ, axis=1)
    yHat = expZ/np.vstack(sigmaExpZ)
    gradient = np.dot(Images.T, (yHat - y)) / n
    return Weights - (EPSILON * gradient)

def softmaxRegression (trainingData, trainingLabels, epsilon = None, batchSize = None, numEpochs = 1):
    if batchSize is None:
        batchSize = trainingLabels.size

    # start with a random weights
    w = 0.001 * np.random.randn(trainingData.shape[1],2)

    numBatches = int(trainingLabels.size / batchSize) - 1

    for _ in range(numEpochs):
        # reorder images
        newOrder = np.random.shuffle(np.arange(trainingLabels.size))
        trainingLabelsNewOrder = trainingLabels[newOrder]
        trainingImagesNewOrder = trainingData[newOrder]
        # Batches
        for e in range(numBatches):
            trainingImagesBatch = trainingData[e * batchSize:(e + 1) * batchSize]
            trainingLabelsBatch = trainingLabels[e*batchSize:(e+1)*batchSize]
            w = gradientDescentSoftMax(trainingImagesBatch, trainingLabelsBatch, w, EPSILON=epsilon)
    return w

def PC(w, data, y):
    z = np.dot(data, w)
    expZ = np.exp(z)
    sigmaExpZ = np.sum(expZ, axis=1)
    yHat = expZ / np.vstack(sigmaExpZ)
    guess = np.argmax(yHat, axis=1)
    correct = np.equal(guess, y)

    return np.sum(correct)/y.size


if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    SibSp = d.SibSp.to_numpy()

    trainingData = np.array([sex, Pclass,SibSp]).T
    trainingLabels = y


    # Train model using part of homework 3.
    w = softmaxRegression(trainingData, trainingLabels, epsilon = 0.1, batchSize = 50, numEpochs = 500)
    # print(w)
    print("Training Accuracy" + str(PC(w, trainingData, trainingLabels)))

    # Load testing data
    d = pandas.read_csv("test.csv")
    sex = d.Sex.map({"male": 0, "female": 1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    SibSp = d.SibSp.to_numpy()
    testingData = np.array([sex, Pclass, SibSp]).T
    PassengerIds = d.PassengerId.to_numpy()

    # Compute predictions on test set
    z = np.dot(testingData, w)
    expZ = np.exp(z)
    sigmaExpZ = np.sum(expZ, axis=1)
    yHat = expZ / np.vstack(sigmaExpZ)
    prediction = np.argmax(yHat, axis=1)

    # Write CSV file of the format:
    # PassengerId, Survived
    df = pandas.DataFrame({'PassengerId': PassengerIds,
                           'Survived': prediction})
    df.to_csv('predictions.csv',index=False)