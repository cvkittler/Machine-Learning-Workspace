import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from datetime import datetime

# this takes in a vector of ground-truth labels and corresponding vector of guesses, and
# then computes the accuracy (PC). The implementation (in vectorized form) should only take 1-line
def fPercentageCorrect(y, yHat):
    return 1 - np.sum(y != yHat)/y.size


# this takes in a set of predictors, a set of images
# to run it on, as well as the ground-truth labels of that set. For each image in the image set, it runs the
# ensemble to obtain a prediction. Then, it computes and returns the accuracy (PC) of the predictions
# w.r.t. the ground-truth labels.
def measureAccuracyOfPredictors(predictors, X, y, numImages):
    yHatEnsemble = np.ones([predictors.shape[0], numImages])
    for i in range(predictors.shape[0]):
        coordPairs = predictors[i]
        firstPixels = X[:numImages,coordPairs[0, 0],coordPairs[0, 1]]
        secondPixels = X[:numImages, coordPairs[1, 0], coordPairs[1, 1]]
        yHatEnsemble[i] = np.greater(firstPixels, secondPixels)
    yHatSum = np.sum(yHatEnsemble, axis=0)
    yHat = np.greater(yHatSum * 2, predictors.shape[0])
    return fPercentageCorrect(y[:numImages], yHat)


# I’ve included some visualization code, but otherwise it’s empty. You need to implement the step-wise classification
# described above.
def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, setSizes):
    numFeatures = 7
    show = True
    file = open("data.txt", "a")
    start = datetime.now()
    date = str(start.date())
    time = str(start.time())
    file.write("Trained on (" + date + " " + time + ")\n")
    file.write("n\ttrainingAccuracy\ttestingAccuracy\n")
    file.close()
    for trainingSize in setSizes:

        imageSize = trainingFaces[0].shape[1]
        predictors = np.zeros([1,2,2], type(np.int32))
        curBestCords = np.array([[0, 0], [0, 0]])
        curBestPercentageCorrect = 0

        for curStep in range(numFeatures):
            for c1 in range(imageSize):
                for r1 in range(imageSize):
                    for c2 in range(imageSize):
                        for r2 in range(imageSize):
                            predictors[curStep,:] = np.array([[c1, r1], [c2, r2]])
                            thisPercentageCorrect = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels, trainingSize)
                            if curBestPercentageCorrect < thisPercentageCorrect:
                                curBestPercentageCorrect = thisPercentageCorrect
                                curBestCords = [[c1, r1], [c2, r2]]
                print("c1: " + str(c1) + " Step: " + str(curStep) + " Training size: " + str(trainingSize) + " Current best PC: " + str(curBestPercentageCorrect))
            predictors[curStep,:,:] = np.array([curBestCords[0],curBestCords[1]])
            if show:
                print("Training Size: " + str(trainingSize) + " Step: " + str(curStep +1))
                # Show an arbitrary test image in grayscale
                im = testingFaces[0, :, :]
                fig, ax = plt.subplots(1)
                ax.imshow(im, cmap='gray')
                # Show r1,c1
                rect = patches.Rectangle((curBestCords[0][0] - 0.5, curBestCords[0][1] - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Show r2,c2
                rect = patches.Rectangle((curBestCords[1][0] - 0.5, curBestCords[1][1] - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                # Display the merged result
                plt.suptitle("Step: " + str(curStep) + " Training size: " + str(trainingSize))
                plt.show()
            if curStep != numFeatures-1:
                curBestCords = [[0, 0], [0, 0]]
                curBestPercentageCorrect = 0
                predictors = np.append(predictors, curBestCords)
            predictors = predictors.reshape(-1, 2, 2)
        file = open("data.txt", "a")
        file.write(str(trainingSize) + " \t" + str(measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels,trainingSize)) + "\t" + str(measureAccuracyOfPredictors(predictors, testingFaces, testingLabels,testingLabels.size))+"\n")
        file.close()
        if show:
            print("Training Size: " + str(trainingSize) + " Step: " + str(curStep + 1))
            # Show an arbitrary test image in grayscale
            im = testingFaces[0, :, :]
            fig, ax = plt.subplots(1)
            ax.imshow(im, cmap='gray')
            for curBestCords in predictors:
                # Show r1,c1
                rect = patches.Rectangle((curBestCords[0][0] - 0.5, curBestCords[0][1] - 0.5), 1, 1, linewidth=2,
                                         edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Show r2,c2
                rect = patches.Rectangle((curBestCords[1][0] - 0.5, curBestCords[1][1] - 0.5), 1, 1, linewidth=2,
                                         edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            # Display the merged result
            plt.suptitle("Step: " + str(curStep) + " Training size: " + str(trainingSize))
            plt.show()

    file = open("data.txt", "a")
    end = datetime.now()
    file.write("Run time: " + str(end - start) + "\n \n \n")
    file.close()

def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    trainingSetSizes = np.array([1])
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, trainingSetSizes)
