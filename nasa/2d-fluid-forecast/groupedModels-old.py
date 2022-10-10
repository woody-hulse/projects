import logging
import warnings
import os
import glob
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from numpy.linalg import inv
import modred as mr

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import finitevolume as sim

# disable matplotlib logger
logging.getLogger('matplotlib.font_manager').disabled = True


def setLogger(showData=True):
    """
    creates debug logger
    :param showData:        show time, type data
    :return:                logger
    """

    if showData:
        logger = logging.getLogger("1")
        handler = logging.StreamHandler()
        formatter1 = logging.Formatter("%(asctime)-30s %(message)s")
        handler.setFormatter(formatter1)
    else:
        logger = logging.getLogger("2")
        handler = logging.StreamHandler()
        formatter2 = logging.Formatter("%(message)s")
        handler.setFormatter(formatter2)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger


debugLogger = setLogger(True)
delimiterLogger = setLogger(False)


def clearDirectory(name):
    """
    clears all files in a directory
    :param name:            name of the directory
    :return:
    """

    name = "./" + name
    for file in os.listdir(name):
        os.remove(os.path.join(name, file))


def saveSim(filename, featureNames, features, numFeatures, resolution, snapshots, time, timestep):
    """
    save data from simulation to a .txt file
    :param filename:        name of save file
    :param featureNames:    list of names of datatypes collected [rho, x velocity, y velocity, pressure]
    :param features:        datasets
    :param numFeatures:     number of datatypes collected
    :param resolution:      number of cells per row/column in data
    :param snapshots:       number of total snapshots in dataset
    :param time:            time multiplier of data
    :return:                none
    """

    with open("savedSimulations/" + filename, "w") as outFile:

        outFile.write(str(numFeatures) + "\n")
        outFile.write(str(resolution) + "\n")
        outFile.write(str(snapshots) + "\n")
        outFile.write(str(time) + "\n")
        outFile.write(str(timestep) + "\n")

        for i in range(numFeatures):
            outFile.write(featureNames[i] + "\n")
            for snapshot in range(len(features[i])):
                for x in range(len(features[i][snapshot])):
                    for y in range(len(features[i][snapshot][x])):
                        outFile.write(str(round(features[i][snapshot][x][y], 4)) + " ")
                    outFile.write("\n")

    debugLogger.debug("saved data to file " + filename)


def loadSim(filename):
    """
    retrieves data from save file
    :param filename:        save file name
    :return:                data from each feature, shape=(features, snapshots, x, y)
                            number of snapshots,
                            resolution (number of cells per row/column in dataset)
                            time multiplier of data
    """

    featureNames, features, numFeatures, resolution, snapshots, time = [], [], 0, 0, 0, 0

    rhoData = []
    vxData = []
    vyData = []
    PData = []

    with open("savedSimulations/" + filename, "r") as inFile:
        numFeatures = int(inFile.readline())
        resolution = int(inFile.readline())
        snapshots = int(inFile.readline())
        time = int(inFile.readline())
        timestep = float(inFile.readline())

        for i in range(numFeatures):
            featureName = inFile.readline().strip()
            featureNames.append(featureName)

            for snapshot in range(snapshots):
                snapshotData = []
                for x in range(resolution):
                    xData = []
                    line = inFile.readline().split()
                    for val in line:
                        xData.append(float(val))
                    snapshotData.append(xData)

                if featureName == "rho":
                    rhoData.append(snapshotData)
                elif featureName == "x velocity":
                    vxData.append(snapshotData)
                elif featureName == "y velocity":
                    vyData.append(snapshotData)
                elif featureName == "pressure":
                    PData.append(snapshotData)

    debugLogger.debug("loaded simulation from file " + filename)

    return rhoData, vxData, vyData, PData, snapshots, resolution, time, timestep


def reorganizeData(data):
    """
    :param data:        input data with shape (features, modes, snapshots)
    :return:            output data with shape (snapshots, modes, features)
    """

    newData = []

    for s in range(len(data[0][0])):
        snapshotData = []
        for m in range(len(data[0])):
            modeData = []
            for f in range(len(data)):
                modeData.append(data[f][m][s])
            snapshotData.append(modeData)
        newData.append(snapshotData)

    newData = np.asarray(newData)

    return newData


def unpackData(data):
    """
    :param data:        input data with shape (snapshots, modes, features)
    :return:            output data with shape (features, modes, snapshots)
    """

    newData = []

    for f in range(len(data[0][0])):
        featureData = []
        for m in range(len(data[0])):
            modeData = []
            for s in range(len(data)):
                modeData.append(data[s][m][f])
            featureData.append(modeData)
        newData.append(featureData)

    newData = np.asarray(newData)

    return newData


def createModel(numModes, resolution, numFeatures, lookback, summary=True):
    """
    create an lstm neural network (four-dimensional) to accomidate multi-featured, two-dimensional data
    :param numModes:            number of temporal modes
    :param resolution:          resolution of data
    :param numFeatures:         number of features [rho, x velocity, y velocity, pressure] in dataset
    :param lookback:            lookback window of model
    :param summary:             boolean to print model information
    :return:
    """

    """
    inputShape = (lookback, numModes, numFeatures)
    outputShape = (numModes, numFeatures)
    """

    inputShape = (lookback, 1)
    outputShape = (1,)

    InputLayer = layers.Input(shape=inputShape)
    Convolution1DLayer1 = layers.Convolution1D(64, 3, input_shape=inputShape, padding="same")(InputLayer)
    MaxPooling1DLayer1 = layers.MaxPooling1D(pool_size=2)(Convolution1DLayer1)
    LSTMLayer = layers.LSTM(100, return_sequences=True)(MaxPooling1DLayer1)
    Convolution1DLayer2 = layers.Convolution1D(32, 3, padding='same')(LSTMLayer)
    MaxPooling1DLayer2 = layers.MaxPooling1D(pool_size=2)(Convolution1DLayer2)
    FlattenLayer = layers.Flatten()(MaxPooling1DLayer2)
    DenseLayer = layers.Dense(1, activation="tanh")(FlattenLayer)
    OutputLayer = layers.Reshape(outputShape)(DenseLayer)

    """
    InputLayer = layers.Input(shape=inputShape)
    LSTMLayer = layers.TimeDistributed(layers.LSTM(4, return_sequences=True))(InputLayer)
    FlatteningLayer1 = layers.Flatten(input_shape=inputShape)(LSTMLayer)
    DenseLayer1 = layers.Dense(64)(FlatteningLayer1)
    DropoutLayer1 = layers.Dropout(0.2)(DenseLayer1)
    DenseLayer2 = layers.Dense(64)(DropoutLayer1)
    DropoutLayer2 = layers.Dropout(0.2)(DenseLayer2)
    DenseLayer3 = layers.Dense(numModes * numFeatures, activation='tanh')(DropoutLayer2)
    OutputLayer = layers.Reshape(outputShape)(DenseLayer3)
    """

    """
    InputLayer = layers.Input(shape=inputShape)
    LSTMLayer = layers.TimeDistributed(layers.LSTM(4, return_sequences=True, activation="relu"))(InputLayer)
    DenseLayer1 = layers.Dense(numModes, activation="relu")(LSTMLayer)
    DenseLayer2 = layers.Dense(numModes)(DenseLayer1)
    Convolutional1DLayer1 = layers.Convolution1D(numModes, 3, 1, padding="valid", activation="relu")(DenseLayer2)
    Convolutional1DLayer2 = layers.Convolution1D(numModes, 3, 1, padding="valid", activation="relu")(Convolutional1DLayer1)
    FlatteningLayer1 = layers.Flatten(input_shape=inputShape)(Convolutional1DLayer2)
    DenseLayer3 = layers.Dense(numModes, activation="relu")(FlatteningLayer1)
    DenseLayer4 = layers.Dense(numModes * numFeatures)(DenseLayer3)
    OutputLayer = layers.Reshape(outputShape)(DenseLayer4)
    """



    model = keras.Model(InputLayer, OutputLayer)
    model.compile(loss='mse', optimizer='adam')

    if summary:
        model.summary()

    # debugLogger.debug("model created\n")

    return model


def trainModel(
        inputTraining,
        outputTraining,
        predictionLength,
        numModes,
        resolution,
        numFeatures,
        lookback,
        epochs,
        train,
        load,
        save,
        plot=True
):
    """
    loads/generates/saves and trains lstm model
    :param inputTraining:       input training dataset, shape=(features, snapshots, modes)
    :param outputTraining:      output training dataset, shape=(features, snapshots, modes)
    :param predictionLength:    length of testing dataset
    :param numModes:            number of temporal modes
    :param resolution:          resolution of data
    :param numFeatures:         number of features [rho, x velocity, y velocity, pressure] in dataset
    :param lookback:            lookback window of model
    :param epochs:              number of training rounds
    :param train:               boolean to train model before fitting
    :param load:                boolean to load model from file
    :param save:                boolean to save model to file
    :param plot:                boolean to plot the loss/epoch data
    :return:                    predicted feature dataset from inputTesting set, shape=(features, snapshots, x, y)
    """

    groupname = "model" + str(numFeatures) + "_" + str(numModes) + "/"
    filepath = "groupedModels/" + groupname

    model_specifications = createModel(numModes, resolution, numFeatures, lookback, True)

    batchSize = 16

    featurePredictedTemporalModes = []
    trainingHistory = []
    for f in range(numFeatures):
        predictedTemporalModes = []
        for m in range(numModes):

            filename = str(f) + "_" + str(m)
            thisfilepath = filepath + filename
            debugLogger.debug("model " + filename)

            if load:
                model = load_model(thisfilepath)
                debugLogger.debug("model loaded from file " + thisfilepath)
            else:
                model = createModel(numModes, resolution, numFeatures, lookback, False)

            if train:
                trainingHistory.append(model.fit(
                    inputTraining[f][m],
                    outputTraining[f][m],
                    epochs=epochs,
                    batch_size=batchSize,
                    verbose=1
                ).history["loss"])

                if save:
                    save_model(model, thisfilepath)

                    debugLogger.debug("model saved to " + thisfilepath)

            predictionData = [inputTraining[f][m][-1]]
            predicted = outputTraining[f][m][-lookback:]

            for i in range(predictionLength):

                modelPrediction = np.reshape(model.predict(np.reshape(predictionData[-1], (1, lookback))), (1,))
                nextPredictionData = np.concatenate((predicted[-lookback + 1:], modelPrediction))
                nextPredictionData = np.reshape(nextPredictionData, (1, lookback))

                predictionData = np.concatenate((predictionData, nextPredictionData))
                predicted = np.concatenate((predicted, modelPrediction))

            predicted = predicted[lookback:]
            predictedTemporalModes.append(predicted)
        featurePredictedTemporalModes.append(predictedTemporalModes)
    featurePredictedTemporalModes = np.array(featurePredictedTemporalModes)

    debugLogger.debug("\n\nforecast data generated from model\n")

    if train and plot:
        for mode in trainingHistory:
            plt.plot(mode)
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.savefig("modelTraining")
        plt.show()

    return featurePredictedTemporalModes


def meanSubtractData(data, snapshots, xDimensions, yDimensions):
    """
    UNUSED: subtract data from common dataset means
    :param data:                single feature data, shape=(features, snapshots, x, y)
    :param snapshots:           number of total snapshots in dataset
    :param xDimensions:         number of cells in x dimension of data
    :param yDimensions:         number of cells in y dimension of data
    :return:                    mean-subtracted dataset, mean of data
    """

    mean = 0
    for i in range(snapshots):
        for j in range(xDimensions):
            for k in range(yDimensions):
                mean += data[i][j][k]

    mean /= xDimensions * yDimensions * snapshots

    for i in range(snapshots):
        for j in range(xDimensions):
            for k in range(xDimensions):
                data[i][j][k] -= mean

    debugLogger.debug("mean subtracted dataset")

    return data, mean


def linearize(data):
    """
    changes (n, n) matrix into (n*n) matrix
    :param data:                (n, n) matrix
    :return:                    (n*n)  matrix
    """

    listData = []

    for row in data:
        for num in row:
            listData.append(num)

    return listData


def undoLinearize(listData, resolution):
    """
    changes (n*n) matrix into (n, n) matrix
    :param listData:                (n*n) matrix
    :param resolution:              n
    :return:                        (n, n) matrix
    """

    data = []
    for i in range(resolution):
        data.append([])
        for j in range(resolution):
            data[-1].append(listData[i * resolution + j])

    return np.array(data)


def diagonalize(data):
    """
    transforms linear data (N) into diagonal matrix (N, N)
    :param data:            input data
    :return:                diagonal matrix
    """

    diagonalized = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            if j == i:
                row.append(data[i])
            else:
                row.append(0)
        diagonalized.append(row)
    diagonalized = np.array(diagonalized)

    return diagonalized


def POD(data, trainingLength, optimalModes=300):

    """
    computes POD spatial modes and temporal coefficients
    :param data:                single feature data
    :param trainingLength:      length of training data
    :param optimalModes:        number of optimal temporal coefficients to use
    :return:                    spatial modes, temporal modes
    """

    reshapedData = []
    for snapshot in data[:trainingLength]:
        reshapedData.append(linearize(snapshot))
    reshapedData = np.array(reshapedData)

    u, s, v = np.linalg.svd(reshapedData.T, full_matrices=True)
    e = s[optimalModes + 1] / s[optimalModes]
    s = diagonalize(s)

    s = s[:optimalModes, :optimalModes]
    v = v[:, :optimalModes]

    spatialModes = u[:, :optimalModes]
    temporalModes = np.matmul(s, v.T)

    debugLogger.debug("computed pod: spatial and temporal modes")
    debugLogger.debug("   tolerance: \u03B5 = " + str(e))

    ##rrmse = np.linalg.norm(np.subtract(v, np.matmul(spatialModes, temporalModes))) / np.linalg.norm(v)
    ##debugLogger.debug("    pod relative root mean squared error: RRMSE = " + str(rrmse))

    return spatialModes, temporalModes


def getCorrectTemporalModes(data, spatialModes):

    """
    gets correct temporal modes from data and spatial modes
    :param data:            single feature data
    :param spatialModes:    spatial modes
    :return:                correct temporal modes
    """

    reshapedData = []
    for snapshot in data:
        reshapedData.append(linearize(snapshot))
    reshapedData = np.array(reshapedData).T

    temporalModes = np.linalg.lstsq(spatialModes, reshapedData, rcond=None)[0]

    return temporalModes


def plotTemporalModes(predictedTemporalModes, correctTemporalModes, numPlots=8, trainingLength=-1):

    numModes = len(predictedTemporalModes)

    if numModes >= 2:
        cols = 2
        rows = math.floor(numPlots / 2)
        figure, axis = plt.subplots(rows, cols)

        for i in range(rows):
            for j in range(cols):
                modeNum = i + j * rows
                axis[i, j].set_title("mode " + str(modeNum))

                if trainingLength != -1:
                    dataLen = len(correctTemporalModes[modeNum])
                    predLen = len(predictedTemporalModes[modeNum])
                    axis[i, j].axvline(x=trainingLength, color='b', linestyle='dashed', label='prediction start')
                    axis[i, j].plot(range(dataLen - predLen, dataLen), predictedTemporalModes[modeNum], 'y', label="predicted")
                else:
                    axis[i, j].plot(predictedTemporalModes[modeNum], 'y', label="predicted")
                axis[i, j].plot(correctTemporalModes[modeNum], 'g', label="correct")
    else:
        plt.title("mode 0")

        if trainingLength != -1:
            dataLen = len(correctTemporalModes[0])
            predLen = len(predictedTemporalModes[0])
            plt.axvline(x=trainingLength, color='b', linestyle='dashed', label='prediction start')
            plt.plot(range(dataLen - predLen, dataLen), predictedTemporalModes[0], 'y', label="predicted")
        else:
            plt.plot(predictedTemporalModes[0], 'y', label="predicted")
        plt.plot(correctTemporalModes[0], 'g', label="correct")

    plt.legend()
    plt.show()



def scaleData(data, scalar=0, inverse=False):
    """
    normalizes and un-normalizes data
    :param data:                data to normalize/un-normalize
    :param scalar:              factor to un-normalize by
    :param inverse:             boolean to un-normalize
    :return:                    data (and scalar if not inverse)
    """

    data = np.asarray(data)

    if not inverse:
        for index, val in np.ndenumerate(data):
            if abs(val) > scalar:
                scalar = abs(val)
        for index, val in np.ndenumerate(data):
            data[index] = val / (2 * scalar) + 0.5

        return data, scalar

    else:
        for index, val in np.ndenumerate(data):
            data[index] = (val - 0.5) * scalar * 2

        return data


def differentiateModes(temporalModes):
    """
    returns a differentiated list of temporal modes
    :param temporalModes:       temporal modes
    :return:                    d/dt
    """

    differentiatedFeatureTemporalModes = []
    for f in range(len(temporalModes)):
        differentiatedTemporalModes = []
        for m in range(len(temporalModes[f])):
            differentiatedMode = [0]
            for s in range(len(temporalModes[f][m]) - 1):
                differentiatedMode.append(temporalModes[f][m][s+1] - temporalModes[f][m][s])
            differentiatedTemporalModes.append(differentiatedMode)
        differentiatedFeatureTemporalModes.append(differentiatedTemporalModes)
    differentiatedFeatureTemporalModes = np.array(differentiatedFeatureTemporalModes)

def reconstructDifferentiation(temporalModes, differentiatedTemporalModes):
    """
    re-integrates differential data into modes
    :param temporalModes:
    :param differentiatedTemporalModes:
    :return:
    """
    featureReconstructedModes = []
    for f in range(len(temporalModes)):
        reconstructedModes = []
        for m in range(len(temporalModes[f])):
            reconstructedMode = list(temporalModes[f][m])
            for s in range(len(differentiatedTemporalModes[f][m])):
                reconstructedMode.append(temporalModes[f][m][-1] + differentiatedTemporalModes[f][m][s])
            reconstructedModes.append(reconstructedMode)
        featureReconstructedModes.append(reconstructedModes)
    featureReconstructedModes = np.array(featureReconstructedModes)


def createTrainingData(temporalModes, trainingLength, numModes, lookback):
    """
    partitions data into training and testing sections at 4:1 ratio, offsets output dataset by 1
    :param temporalModes:       temporal modes of data
    :param trainingLength:      number of snapshots in training data
    :param numModes:            number of temporal modes
    :param lookback:            lookback window of data
    :return:                    input/output training dataset
    """


    featureInputTrainingTemporalModes, featureOutputTrainingTemporalModes = [], []
    for f in range(len(temporalModes)):
        inputTrainingTemporalModes, outputTrainingTemporalModes = [], []
        for m in range(len(temporalModes[0])):
            inputMode, outputMode = [], []
            for i in range(lookback, trainingLength):
                inputMode.append(temporalModes[f][m][i-lookback:i])
                outputMode.append(temporalModes[f][m][i])
            inputTrainingTemporalModes.append(inputMode)
            outputTrainingTemporalModes.append(outputMode)
        featureInputTrainingTemporalModes.append(inputTrainingTemporalModes)
        featureOutputTrainingTemporalModes.append(outputTrainingTemporalModes)

    featureInputTrainingTemporalModes = np.array(featureInputTrainingTemporalModes)
    featureOutputTrainingTemporalModes = np.array(featureOutputTrainingTemporalModes)

    debugLogger.debug("\n\ncreated training and testing datasets\n")

    return featureInputTrainingTemporalModes, \
           featureOutputTrainingTemporalModes


def reconstructData(spatialModes, temporalModes, resolution, scalar):

    """
    reconstruct data from predicted temporal coefficients
    :param spatialModes:        spatial modes of data
    :param temporalModes:       predicted temporal modes of data
    :param resolution:          resolution of data
    :param scalar:              scalar to reverse normalization
    :return:                    reconstructed predicted data
    """

    reconstructedData = np.matmul(spatialModes, temporalModes).T
    reconstructedData = [undoLinearize(snap, resolution) for snap in reconstructedData]
    reconstructedData = np.array(reconstructedData)

    return reconstructedData


def calculateMeanSquaredError(predictedData, correctData, xDimensions, yDimensions, numFeatures):
    """
    calculates the error of data by totaling the squared differences of each predicted datapoint from its corresponding
    point in the correct dataset and dividing by the total number of datapoints in the set
    :param predictedData:       reconstructed multi-feature predicted data
    :param correctData:         correct multi-feature data
    :param xDimensions:         number of cells in x dimension of data
    :param yDimensions:         number of cells in y dimension of data
    :param numFeatures:         number of features represented in dataset
    :return:                    list of mean squared loss for each feature
    """

    featureMeanSquaredErrors = []
    numData = min(len(predictedData), len(correctData)) * xDimensions * yDimensions

    for f in range(numFeatures):
        meanSquaredErrors = []
        for snapshot in range(min(len(predictedData[f]), len(correctData[f]))):
            squaredResidual = 0
            for x in range(xDimensions):
                for y in range(yDimensions):
                    squaredResidual += math.pow(predictedData[f][snapshot][x][y] - correctData[f][snapshot][x][y], 2)
            meanSquaredErrors.append(squaredResidual / numData)
        featureMeanSquaredErrors.append(meanSquaredErrors)

    return featureMeanSquaredErrors


def interval2D(data, resolution, spacing):
    """
    partitions 2-dimensional data into interval grid of data
    :param data:                2 dimensional data to partition
    :param resolution:          number of cells in dataset rows/columns
    :param spacing:             desired spacing of datapoints
    :return:                    interval 2D data
    """

    newData = []
    for i in range(math.floor(resolution / spacing)):
        newData.append([])
        for j in range(math.floor(resolution / spacing)):
            newData[i].append(data[i * spacing][j * spacing])

    return newData


def plotData(datasets, title, xName, yName, labels):
    """
    plots 2-dimensional line data
    :param datasets:            multi-featured linear y data to plot
    :param title:               graph title
    :param xName:               graph x axis label
    :param yName:               graph y axis label
    :param labels:              graph legend labels
    :return:                    none
    """

    xValues = [i for i in range(len(datasets[0]))]
    plt.figure(figsize=(8, 8))
    plt.title(title)
    for i in range(len(datasets)):
        plt.plot(xValues, datasets[i], label=labels[i])
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.legend()
    plt.show()


def plotFlow(featurePredictedData, featureCorrectData, featureNames, numFeatures, start, interval, resolution,
             vxvy=True):
    """
    plots 2-dimensional pressure, flow of each feature of dataset in heatmaps and vector fields
    :param featurePredictedData:    multi-featured predicted data to plot
    :param featureCorrectData:      multi-featured correct data to plot
    :param featureNames:            indexed names of each feature in dataset
    :param numFeatures:             number of features in dataset
    :param interval:                snapshot interval at which plots should be displayed
    :param resolution:              number of cells in each row/column of data
    :param vxvy:                    boolean to combine the x and y flow fields into one
    :return:                        none
    """

    resultsDirectory = "modelResults"
    clearDirectory(resultsDirectory)

    if vxvy and "x velocity" in featureNames and "y velocity" in featureNames:
        vxIndex = featureNames.index("x velocity")
        vyIndex = featureNames.index("y velocity")
        pvxy = [featurePredictedData[vxIndex], featurePredictedData[vyIndex]]
        cvxy = [featureCorrectData[vxIndex], featureCorrectData[vyIndex]]

        featurePredictedData = featurePredictedData[:vxIndex] + \
                               featurePredictedData[vxIndex + 1:vyIndex] + \
                               featurePredictedData[vyIndex + 1:]
        featureCorrectData = featureCorrectData[:vxIndex] + \
                             featureCorrectData[vxIndex + 1:vyIndex] + \
                             featureCorrectData[vyIndex + 1:]
        featureNames = featureNames[:vxIndex] + featureNames[vxIndex + 1:vyIndex] + featureNames[vyIndex + 1:]

        featurePredictedData.append(pvxy)
        featureCorrectData.append(cvxy)
        featureNames.append("velocity")
        numFeatures -= 1

    dataLength = len(featurePredictedData[0])
    featureData = [featurePredictedData, featureCorrectData]
    titles = ["predicted", "correct"]

    saveLimit = 400
    saveIndex = 0

    for i in range(numFeatures):
        for j in range(start, dataLength, (dataLength - start) // 10):
            saveIndex += 1
            if saveIndex > saveLimit:
                break
            plt.cla()
            for k in range(2):
                plt.subplot(1, 2, k + 1)
                plt.title(titles[k] + " " + featureNames[i] + ": t = " + str(j))
                ax = plt.gca()
                if featureNames[i] == "rho" or featureNames[i] == "pressure":
                    plt.imshow(featureData[k][i][j].T)
                    ax.invert_yaxis()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                elif featureNames[i] == "x velocity" or featureNames[i] == "y velocity" or featureNames[
                    i] == "velocity":
                    vectorSpacing = int(resolution / min(16, resolution / 2))
                    x = [m for m in range(0, resolution, vectorSpacing)]
                    y = [n for n in range(0, resolution, vectorSpacing)]
                    if featureNames[i] == "x velocity":
                        xIntervalData = interval2D(featureData[k][i][j], resolution, vectorSpacing)
                        zeros = np.zeros(np.shape(xIntervalData))
                        plt.quiver(x, y, xIntervalData, zeros)
                    elif featureNames[i] == "y velocity":
                        yIntervalData = interval2D(featureData[k][i][j], resolution, vectorSpacing)
                        zeros = np.zeros(np.shape(yIntervalData))
                        plt.quiver(x, y, zeros, yIntervalData)
                    elif featureNames[i] == "velocity":
                        xIntervalData = interval2D(featureData[k][i][0][j], resolution, vectorSpacing)
                        yIntervalData = interval2D(featureData[k][i][1][j], resolution, vectorSpacing)
                        plt.quiver(x, y, xIntervalData, yIntervalData)
                ax.set_aspect("equal")
            plt.savefig(resultsDirectory + "/plot_" + str(saveIndex) + ".png")
            plt.close()


def animatePlots():
    """
    animates plots in modelResults output directory
    :return:
    """

    gif_name = "flow"
    file_list = glob.glob("./modelResults/*.png")
    list.sort(file_list, key=lambda x: int(x.split("_")[1].split(".png")[0]))

    with open("image_list.txt", "w") as file:
        for item in file_list:
            file.write("%s\n" % item)

    os.system("convert @image_list.txt {}.gif".format(gif_name))


def plotCrossSection(data1, data2, index):
    row = int(len(data1[0]) / 2)

    datasets = [data1[index][row], data2[index][row]]

    plotData(datasets, "Cross section of mode " + str(index + 1), "col", "val", ["predicted", "correct"])


def main(train=False,
         load=True,
         save=False,
         saveSimulation=False,
         loadSimulation=False,
         epochs=4,
         resolution=64,
         numModes=28,
         time=2,
         timestep=0.01,
         plot=True,
         ):

    """

    Fluid Prediction via Long-Short Term Memory Neural Network Modeling

    :param train:           boolean to train model
    :param load:            boolean to load model
    :param save:            boolean to save model data
    :param saveSimulation:  boolean to save simulation data
    :param loadSimulation:  boolean to load simulation data from file
    :param epochs:          number of training epochs (if train)
    :param resolution:      number of cells in each row/column of data
    :param numModes  number of spatial modes and temporal modes
    :param time:            time multiplier of data
    :param timestep:        dt
    :param plot:            boolean to plot predicted data
    :return:                none
    """

    filename = "sim" + str(resolution) + "_" + str(time) + "_" + str(timestep) + ".txt"

    if loadSimulation:
        rhoData, vxData, vyData, PData, snapshots, resolution, time, timestep = loadSim(filename)
    else:
        rhoData, vxData, vyData, PData, snapshots = sim.main(resolution, time, timestep)

    xDimensions, yDimensions = len(rhoData[0]), len(rhoData[0][0])

    # available features: rho, x velocity, y velocity, pressure
    featureNames = ["rho", "x velocity", "y velocity", "pressure"]
    features = []
    for name in featureNames:
        if name == "rho":
            features.append(rhoData)
        elif name == "x velocity":
            features.append(vxData)
        elif name == "y velocity":
            features.append(vyData)
        elif name == "pressure":
            features.append(PData)
    numFeatures = len(features)

    if saveSimulation and not loadSimulation:
        saveSim(filename, featureNames, features, numFeatures, resolution, snapshots, time, timestep)

    featureTemporalModes = []
    featureSpatialModes = []
    featureScalars = []
    trainingRatio = 0.8
    lookback = 8 # keep

    trainingLength = int(trainingRatio * snapshots)
    testingLength = snapshots - trainingLength

    featureIndex = 0
    for data in features:
        delimiterLogger.debug(featureNames[featureIndex])
        featureIndex += 1

        spatialModes, temporalModes = POD(data, trainingLength, numModes)
        correctTemporalModes = getCorrectTemporalModes(data[:trainingLength], spatialModes)
        temporalModes = correctTemporalModes
        temporalModes, scalar = scaleData(temporalModes)

        featureTemporalModes.append(temporalModes)
        featureSpatialModes.append(spatialModes)
        featureScalars.append(scalar)

    featureTemporalModes = np.array(featureTemporalModes)

    featureInputTrainingTemporalModes, featureOutputTrainingTemporalModes = createTrainingData(
        featureTemporalModes,
        trainingLength,
        numModes,
        lookback
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        featurePredictedTemporalModes = trainModel(
            np.array(featureInputTrainingTemporalModes),
            np.array(featureOutputTrainingTemporalModes),
            testingLength,
            numModes,
            resolution,
            numFeatures,
            lookback,
            epochs,
            train,
            load,
            save
        )

        viewModes = False
        if viewModes:
            for f in range(numFeatures):
                plt.cla()
                plt.imshow(featurePredictedTemporalModes[f][10:].T)
                ax = plt.gca()
                ax.invert_yaxis()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect('equal')

    featureCorrectTemporalModes = []
    for f in range(numFeatures):
        featureCorrectTemporalModes.append(getCorrectTemporalModes(features[f], featureSpatialModes[f]))
        featureTemporalModes[f] = scaleData(featureTemporalModes[f], featureScalars[f], True)
        featurePredictedTemporalModes[f] = scaleData(featurePredictedTemporalModes[f], featureScalars[f], True)
    featureCorrectTemporalModes = np.array(featureCorrectTemporalModes)

    featurePredictedData = []
    featureCombinedTemporalModes = np.concatenate((featureTemporalModes, featurePredictedTemporalModes), axis=2)
    plotTemporalModes(featurePredictedTemporalModes[0], featureCorrectTemporalModes[0], 4, trainingLength)
    totalFeatureTemporalModes = np.array(featureCombinedTemporalModes)
    featureCorrectData = features
    for f in range(numFeatures):
        predictedData = reconstructData(
            featureSpatialModes[f],
            totalFeatureTemporalModes[f],
            resolution,
            featureScalars[f]
        )
        featurePredictedData.append(predictedData)

    delimiterLogger.debug("number of snapshots: " + str(snapshots))
    delimiterLogger.debug("length of prediction: " + str(testingLength))

    featureErrors = calculateMeanSquaredError(
        featurePredictedData,
        featureCorrectData,
        xDimensions,
        yDimensions,
        numFeatures
    )

    plotCrossSection(featurePredictedData[0], featureCorrectData[0], 5)

    plotInterval = int(time * 2)
    plotData(featureErrors, "feature errors", "t", "MSE", featureNames)

    if plot:
        plotFlow(
            featurePredictedData,
            featureCorrectData,
            featureNames,
            numFeatures,
            trainingLength,
            plotInterval,
            resolution,
            vxvy=True
        )

        animatePlots()


if __name__ == "__main__":
    saveFile = "sim.txt"

    os.system("clear")

    # train, load, save, save sim, load sim, epochs, resolution, time, plot
    # savefile format is save[resolution]_[time]_[timestep].txt
    main(
        True,  # Train model
        False,  # Load model from file
        True,  # Save model to file
        False,  # Save simulation data to file
        False,  # Load simulation data from file
        256,  # Training epochs
        128,  # Data resolution
        16,  # Number of modes
        2,  # Time multiplier
        0.002,  # Timestep length
        True,  # Plot data
    )
