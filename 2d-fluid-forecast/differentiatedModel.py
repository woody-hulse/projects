import logging
import warnings
import os
import glob
import time
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from numpy.linalg import inv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import figure
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
    [obsolete] save data from simulation to a .txt file
    :param filename:        name of save file
    :param featureNames:    list of names of datatypes collected [rho, u, v, pressure]
    :param features:        datasets
    :param numFeatures:     number of datatypes collected
    :param resolution:      number of cells per row/column in data
    :param snapshots:       number of total snapshots in dataset
    :param time:            time multiplier of data
    :return:                none
    """

    with open(filename, "w") as outFile:

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
    [obsolete] retrieves data from save file
    :param filename:        save file name
    :return:                data from each feature, shape=(features, snapshots, x, y)
                            number of snapshots,
                            resolution (number of cells per row/column in dataset)
                            time multiplier of data
    """

    featureNames, features, numFeatures, resolution, snapshots, time = [], [], 0, 0, 0, 0

    rhoData = []
    uData = []
    vData = []
    PData = []

    with open(filename, "r") as inFile:
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
                elif featureName == "u":
                    uData.append(snapshotData)
                elif featureName == "v":
                    vData.append(snapshotData)
                elif featureName == "P":
                    PData.append(snapshotData)

    debugLogger.debug("loaded simulation from file " + filename)

    return rhoData, uData, vData, PData, snapshots, resolution, time, timestep


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
    :param numFeatures:         number of features [rho, u, v, pressure] in dataset
    :param lookback:            lookback window of model
    :param summary:             boolean to print model information
    :return:
    """

    inputShape = (lookback, numModes, numFeatures)
    outputShape = (numModes, numFeatures)

    """

    InputLayer = layers.Input(shape=inputShape)
    LSTMLayer = layers.TimeDistributed(layers.LSTM(numModes, return_sequences=True))(InputLayer)
    FlatteningLayer1 = layers.Flatten(input_shape=inputShape)(LSTMLayer)
    DenseLayer1 = layers.Dense(numFeatures * numModes)(FlatteningLayer1)
    DropoutLayer1 = layers.Dropout(0.2)(DenseLayer1)
    DenseLayer2 = layers.Dense(numFeatures * numModes)(DropoutLayer1)
    DropoutLayer2 = layers.Dropout(0.2)(DenseLayer2)
    DenseLayer3 = layers.Dense(numFeatures * numModes, activation='tanh')(DropoutLayer2)
    OutputLayer = layers.Reshape(outputShape)(DenseLayer3)

    """

    InputLayer = layers.Input(shape=inputShape)
    ReshapeLayer1 = layers.Reshape((lookback, numModes * numFeatures))(InputLayer)
    LSTMLayer1 = layers.LSTM(100, activation="tanh", return_sequences=True)(ReshapeLayer1)
    LSTMLayer2 = layers.LSTM(100, activation="tanh", return_sequences=True)(LSTMLayer1)
    FlattenLayer1 = layers.Flatten()(LSTMLayer2)
    DenseLayer1 = layers.Dense(100)(FlattenLayer1)
    DenseLayer2 = layers.Dense(numModes * numFeatures)(DenseLayer1)
    OutputLayer = layers.Reshape(outputShape)(DenseLayer2)

    model = keras.Model(InputLayer, OutputLayer)
    model.compile(loss='mse', optimizer='adam')

    if summary:
        model.summary()

    debugLogger.debug("model created\n")

    return model


def trainModel(
        simulation,
        inputTraining,
        outputTraining,
        truncationLength,
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
    :param simulation:          simulation type
    :param inputTraining:       input training dataset, shape=(snapshots, lookback, modes, features)
    :param outputTraining:      output training dataset, shape=(snapshots, modes, features)
    :param truncationLength:    length of data to truncate from beginning of training set
    :param predictionLength:    length of testing dataset
    :param numModes:            number of temporal modes
    :param resolution:          resolution of data
    :param numFeatures:         number of features [rho, u, v, pressure] in dataset
    :param lookback:            lookback window of model
    :param epochs:              number of training rounds
    :param train:               boolean to train model before fitting
    :param load:                boolean to load model from file
    :param save:                boolean to save model to file
    :param plot:                boolean to plot the loss/epoch data
    :return:                    predicted feature dataset from inputTesting set, shape=(features, snapshots, x, y)
    """

    inputTraining = truncateFeatureData(inputTraining, truncationLength)
    outputTraining = truncateFeatureData(outputTraining, truncationLength)

    model = createModel(numModes, resolution, numFeatures, lookback)
    filename = "model" + str(numModes)
    filepath = simulation + "Models/" + filename

    if load:
        model = load_model(filepath)

        debugLogger.debug("model loaded from file " + filepath)

    batchSize = 24

    if train:
        if save:
            if epochs >= 128:
                if epochs >= 512:
                    verbose = 2
                else:
                    verbose = 1
                trainingHistory = model.fit(
                    inputTraining,
                    outputTraining,
                    epochs=epochs,
                    batch_size=batchSize,
                    verbose=verbose
                )
                save_model(model, filepath)

                debugLogger.debug("model saved to " + filepath)

            else:
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    verbose=1,
                    monitor="loss",
                    save_best_only=True
                )
                trainingHistory = model.fit(
                    inputTraining,
                    outputTraining,
                    epochs=epochs,
                    batch_size=batchSize,
                    verbose=1,
                    callbacks=checkpoint
                )
        else:
            trainingHistory = model.fit(
                inputTraining,
                outputTraining,
                epochs=epochs,
                batch_size=batchSize,
                verbose=1
            )

        if plot:
            plt.plot(trainingHistory.history["loss"], label="loss")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("mse")
            plt.savefig("modelTraining")
            plt.show()

    predictionData = [inputTraining[-1]]
    predicted = outputTraining[-lookback:]

    for i in range(predictionLength):

        modelPrediction = model.predict(np.reshape(predictionData[-1], (1, lookback, numModes, numFeatures)))
        nextPredictionData = np.concatenate((predicted[-lookback + 1:], modelPrediction))
        nextPredictionData = np.reshape(nextPredictionData, (1, lookback, numModes, numFeatures))

        predictionData = np.concatenate((predictionData, nextPredictionData))
        predicted = np.concatenate((predicted, modelPrediction))

    try:
        predicted = predicted[lookback:]
        predicted = unpackData(predicted)

    except:
        return np.array([[[]]])

    debugLogger.debug("\n\nforecast data generated from model\n")

    return predicted


def generateControlData1(outputTraining, predictionLength, numModes, numFeatures):
    """
    generates prediction data as a linear continuation of training data
    :param outputTraining:      output training dataset, shape=(features, snapshots, modes)
    :param predictionLength:    length of testing dataset
    :param numModes:            number of temporal modes
    :param numFeatures:         number of features [rho, u, v, pressure] in dataset
    :return:                    control predicted data
    """

    dt = 5 # snapshot length of differentiation sample
    outputTraining = reorganizeData(outputTraining)

    dxdt = np.zeros((1, numModes, numFeatures))
    for f in range(numFeatures):
        for m in range(numModes):
            dxdt[0][m][f] = (outputTraining[-1][m][f] - outputTraining[-1 - dt][m][f]) / dt

    predicted = np.zeros((predictionLength, numModes, numFeatures))
    predicted[0] = outputTraining[-1] + dxdt
    for s in range(1, predictionLength):
        predicted[s] = predicted[s - 1] + dxdt

    predicted = unpackData(predicted)

    return predicted


def generateControlData2(outputTraining, predictionLength, numModes, numFeatures):
    """
    generates prediction data as constant continuation of training data
    :param outputTraining:      output training dataset, shape=(features, snapshots, modes)
    :param predictionLength:    length of testing dataset
    :param numModes:            number of temporal modes
    :param numFeatures:         number of features [rho, u, v, pressure] in dataset
    :return:                    control predicted data
    """

    outputTraining = reorganizeData(outputTraining)

    predicted = np.zeros((predictionLength, numModes, numFeatures))
    predicted[0] = outputTraining[-1]
    for s in range(1, predictionLength):
        predicted[s] = predicted[s-1]

    predicted = unpackData(predicted)

    return predicted


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


def POD(data, trainingLength, numModes=300):

    """
    computes POD spatial modes and temporal coefficients
    :param data:                single feature data
    :param trainingLength:      length of training data
    :param numModes:            number of modes
    :return:                    spatial modes, temporal modes
    """

    reshapedData = []
    for snapshot in data[:trainingLength]:
        reshapedData.append(np.array(snapshot).flatten())
    reshapedData = np.array(reshapedData)

    u, s, v = np.linalg.svd(reshapedData.T, full_matrices=True)
    e = s[numModes + 1] / s[numModes]
    energy = sum(s[:numModes]) / sum(s)
    s = diagonalize(s)

    s = s[:numModes, :numModes]
    v = v[:, :numModes]

    spatialModes = u[:, :numModes]
    temporalModes = np.matmul(s, v.T)

    debugLogger.debug("computed pod: spatial and temporal modes")
    debugLogger.debug("   tolerance: \u03B5 = " + str(e))
    debugLogger.debug("      energy: e = " + str(energy))

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


def differentiateModes(temporalModes):
    """
    returns a differentiated list of temporal modes
    :param temporalModes:       temporal modes
    :return:                    d/dt
    """

    differentiatedTemporalModes = []
    for m in range(len(temporalModes)):
        differentiatedMode = [0]
        for s in range(len(temporalModes[m]) - 1):
            differentiatedMode.append(temporalModes[m][s + 1] - temporalModes[m][s])
        differentiatedTemporalModes.append(differentiatedMode)
    differentiatedTemporalModes = np.array(differentiatedTemporalModes)

    return differentiatedTemporalModes


def reconstructDifferentiation(temporalModes, differentiatedTemporalModes):
    """
    re-integrates differential data into modes
    :param temporalModes:                   temporal modes
    :param differentiatedTemporalModes:     predicted d/dt data
    :return:                                total temporal modes
    """

    reconstructedModes = []
    for m in range(len(temporalModes)):
        reconstructedMode = list(temporalModes[m])
        for s in range(len(differentiatedTemporalModes[m])):
            reconstructedMode.append(reconstructedMode[-1] + differentiatedTemporalModes[m][s])
        reconstructedModes.append(reconstructedMode)
    reconstructedModes = np.array(reconstructedModes)

    return reconstructedModes


def plotTemporalModes(predictedTemporalModes,
                      correctTemporalModes,
                      controlTemporalModes,
                      title="",
                      numPlots=8,
                      trainingLength=-1,
                      truncationLength=0):

    """
    plots temporal modes with predicted data, correct data, and appropriate demarcations
    :param predictedTemporalModes:      predicted temporal modes
    :param correctTemporalModes:        correct temporal modes
    :param controlTemporalModes:        control temporal modes
    :param title:                       title of graph
    :param numPlots:                    number of plots
    :param trainingLength:              length of training data
    :param truncationLength:            length of truncation
    :return:
    """

    numModes = len(predictedTemporalModes)

    cols = 4
    rows = math.floor(numPlots / cols)

    figure, axis = plt.subplots(rows, cols)
    plt.title(title)

    for i in range(rows):
        for j in range(cols):
            modeNum = i + j * rows

            if modeNum < numModes:
                axis[i, j].set_title("mode " + str(modeNum))
                axis[i, j].plot(correctTemporalModes[modeNum], 'g', label="correct")

                if trainingLength != -1:
                    dataLen = len(correctTemporalModes[modeNum])
                    predLen = len(predictedTemporalModes[modeNum])
                    axis[i, j].axvline(x=truncationLength, color='c', linestyle='dashed', label='training start')
                    axis[i, j].axvline(x=trainingLength, color='b', linestyle='dashed', label='prediction start')
                    axis[i, j].plot(range(dataLen - predLen, dataLen), predictedTemporalModes[modeNum], 'y', label="predicted")
                    for controlIndex in range(len(controlTemporalModes)):
                        axis[i, j].plot(range(dataLen - predLen, dataLen),
                                        controlTemporalModes[controlIndex][modeNum],
                                        label="control" + str(controlIndex))

                else:
                    axis[i, j].plot(predictedTemporalModes[modeNum], 'y', label="predicted")

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


def truncateFeatureData(featureData, truncationLength=0):
    """
    truncates first portion of data by truncationLength amount
    :param featureData:         feature data to truncate (snapshots, , modes, features)
    :param truncationLength:    amount to truncate from beginning
    :return:                    truncated data
    """

    truncatedData = featureData[truncationLength:]

    return truncatedData



def createTrainingData(temporalModes, trainingLength, numModes, lookback):
    """
    partitions data into training and testing sections at 4:1 ratio, offsets output dataset by 1
    :param temporalModes:       temporal modes of data
    :param trainingLength:      number of snapshots in training data
    :param numModes:            number of temporal modes
    :param lookback:            lookback window of data
    :return:                    input/output training dataset
    """

    trainingData = reorganizeData(temporalModes)

    inputTrainingTemporalModes, outputTrainingTemporalModes = [], []
    for i in range(lookback, trainingLength):
        inputTrainingTemporalModes.append(trainingData[i-lookback:i])
        outputTrainingTemporalModes.append(trainingData[i])

    inputTrainingTemporalModes = np.array(inputTrainingTemporalModes)
    outputTrainingTemporalModes = np.array(outputTrainingTemporalModes)

    debugLogger.debug("\n\ncreated training and testing datasets\n")

    return inputTrainingTemporalModes, \
           outputTrainingTemporalModes


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


def calculateMeanSquaredError(predictedData, correctData, resolution, numFeatures):
    """
    calculates the error of data by totaling the squared differences of each predicted datapoint from its corresponding
    point in the correct dataset and dividing by the total number of datapoints in the set
    :param predictedData:       reconstructed multi-feature predicted data
    :param correctData:         correct multi-feature data
    :param resolution:          resoluton of data
    :param numFeatures:         number of features represented in dataset
    :return:                    list of mean squared loss for each feature
    """

    featureMeanSquaredErrors = []
    numData = min(len(predictedData), len(correctData)) * resolution**2

    for f in range(numFeatures):
        meanSquaredErrors = []
        for snapshot in range(min(len(predictedData[f]), len(correctData[f]))):
            squaredResidual = 0
            for x in range(resolution):
                for y in range(resolution):
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


def plotFlow(featurePredictedData, featureCorrectData, featureControlDataSets, featureNames, numFeatures, start, interval, resolution):
    """
    plots 2-dimensional pressure, flow of each feature of dataset in heatmaps and vector fields
    :param featurePredictedData:    multi-featured predicted data to plot
    :param featureCorrectData:      multi-featured correct data to plot
    :param featureControlDataSets   multi-feature control datasets to plot
    :param featureNames:            indexed names of each feature in dataset
    :param numFeatures:             number of features in dataset
    :param interval:                snapshot interval at which plots should be displayed
    :param resolution:              number of cells in each row/column of data
    :return:                        none
    """

    resultsDirectory = "modelResults"
    clearDirectory(resultsDirectory)
    featureNames = ["rho", "velocity", "pass", "P",]

    numControls = len(featureControlDataSets)
    numPlots = numControls + 2

    dataLength = len(featurePredictedData[0])
    featureControlDataSets = np.array(featureControlDataSets)
    featureData = np.array([featurePredictedData, featureCorrectData])
    featureData = np.array(featureData)
    featureData = np.concatenate((featureData, featureControlDataSets), axis=0)
    featureData = np.reshape(featureData, (numPlots, numFeatures, dataLength, resolution, resolution))
    titles = ["predicted", "correct"] + ["control" + str(i) for i in range(len(featureControlDataSets))]

    saveLimit = 400
    saveIndex = 0

    for i in range(numFeatures):
        if featureNames[i] == "pass":
            continue
        for j in range(start, dataLength, (dataLength - start) // 50):
            saveIndex += 1
            if saveIndex > saveLimit:
                break
            plt.cla()
            for k in range(numPlots):
                plt.subplot(2, math.ceil(numPlots / 2), k + 1)
                plt.title(titles[k] + " " + featureNames[i] + ": t = " + str(j))
                ax = plt.gca()
                if featureNames[i] == "rho" or featureNames[i] == "P":
                    plt.imshow(featureData[k][i][j].T)
                    ax.invert_yaxis()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                elif featureNames[i] == "u" or featureNames[i] == "v" or featureNames[i] == "velocity":
                    vectorSpacing = int(resolution / min(64, resolution / 2))
                    x = [m for m in range(0, resolution, vectorSpacing)]
                    y = [n for n in range(0, resolution, vectorSpacing)]
                    if featureNames[i] == "u":
                        xIntervalData = interval2D(featureData[k][i][j].T, resolution, vectorSpacing)
                        zeros = np.zeros(np.shape(xIntervalData))
                        plt.quiver(x, y, xIntervalData, zeros)
                    elif featureNames[i] == "v":
                        yIntervalData = interval2D(featureData[k][i][j].T, resolution, vectorSpacing)
                        zeros = np.zeros(np.shape(yIntervalData))
                        plt.quiver(x, y, zeros, yIntervalData)
                    elif featureNames[i] == "velocity":
                        xIntervalData = interval2D(featureData[k][i][j].T, resolution, vectorSpacing)
                        yIntervalData = interval2D(featureData[k][i+1][j].T, resolution, vectorSpacing)
                        plt.quiver(x, y, xIntervalData, yIntervalData)
                ax.set_aspect("equal")
            plt.savefig(resultsDirectory + "/plot_" + str(saveIndex) + ".png")
            plt.close()


def animatePlots(resolution, timestep):
    """
    animates plots in modelResults output directory
    :param resolution:          resolution of data
    :param timestep:            timestep of snapshots
    :return:
    """

    gif_name = "flow"
    file_list = glob.glob("./modelResults/*.png")
    list.sort(file_list, key=lambda x: int(x.split("_")[1].split(".png")[0]))

    with open("image_list.txt", "w") as file:
        for item in file_list:
            file.write("%s\n" % item)

    os.system("convert @image_list.txt {}.gif".format(gif_name))


def plotCrossSection(predictedData, correctData, controls, index):
    row = int(len(predictedData[0]) / 2)

    datasets = [predictedData[index][row], correctData[index][row]] + [controlData[index][row] for controlData in controls]

    plotData(datasets, "Cross section of snapshot " + str(index + 1), "col", "val",
             ["predicted", "correct"] + ["control" + str(i) for i in range(len(controls))])


def main(train=False,
         load=True,
         save=False,
         simulation="kelvinHelmholtz",
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

    :param simulation:      simulation used for model
    :param train:           boolean to train model
    :param load:            boolean to load model
    :param save:            boolean to save model data
    :param saveSimulation:  boolean to save simulation data
    :param loadSimulation:  boolean to load simulation data from file
    :param epochs:          number of training epochs (if train)
    :param resolution:      number of cells in each row/column of data
    :param numModes:        number of spatial modes and temporal modes
    :param time:            time multiplier of data
    :param timestep:        dt
    :param plot:            boolean to plot predicted data
    :return:                none
    """

    filepath = simulation + "/" + str(resolution).zfill(3)  + "/" + str(time).zfill(2) + "_" + str(timestep) + "/"

    features_filename = "features.npy"
    featureTemporalModes_filename = "temporalModes.npy"
    featureDifferentiatedTemporalModes_filename = "differentiatedTemporalModes.npy"
    featureSpatialModes_filename = "spatialModes.npy"
    featureScalars_filename = "scalars.npy"

    # available features: rho, u, v, P
    featureNames = ["rho", "u", "v", "P"]
    truncationRatio = 0
    trainingRatio = 0.8
    lookback = 4

    loadFeaturesOnly = False
    if loadSimulation and not os.path.exists(filepath + str(numModes) + "/"):
        loadFeaturesOnly = True
        loadSimulation = False

    if loadSimulation:
        filepath2 = filepath + str(numModes) + "/"
        features = np.load(filepath + features_filename)
        featureTemporalModes = np.load(filepath2 + featureTemporalModes_filename)
        featureDifferentiatedTemporalModes = np.load(filepath2 + featureDifferentiatedTemporalModes_filename)
        featureSpatialModes = np.load(filepath2 + featureSpatialModes_filename)
        featureScalars = np.load(filepath2 + featureScalars_filename)

        delimiterLogger.debug("\n\nloaded precomputed data from " + filepath + "\n")

        numFeatures = len(features)
        snapshots = len(features[0])
        trainingLength = int(trainingRatio * snapshots)
        truncationLength = int(truncationRatio * snapshots)
        predictionLength = snapshots - trainingLength
    else:
        if loadFeaturesOnly:
            features = np.load(filepath + features_filename)
            delimiterLogger.debug("\n\nloaded data from " + filepath + "\n")
        else:
            rhoData, uData, vData, PData, snapshots = sim.main(resolution, time, timestep)
            features = [rhoData, uData, vData, PData]
        features = np.array(features)

        numFeatures = len(features)
        snapshots = len(features[0])
        trainingLength = int(trainingRatio * snapshots)
        truncationLength = int(truncationRatio * snapshots)
        predictionLength = snapshots - trainingLength

        featureTemporalModes = []
        featureDifferentiatedTemporalModes = []
        featureSpatialModes = []
        featureScalars = []

        featureIndex = 0
        for data in features:
            delimiterLogger.debug(featureNames[featureIndex])
            featureIndex += 1

            spatialModes, temporalModes = POD(data, trainingLength, numModes)
            correctTemporalModes = getCorrectTemporalModes(data[:trainingLength], spatialModes)
            temporalModes = correctTemporalModes
            differentiatedTemporalModes = differentiateModes(temporalModes)
            differentiatedTemporalModes, scalar = scaleData(differentiatedTemporalModes)

            featureTemporalModes.append(temporalModes)
            featureDifferentiatedTemporalModes.append(differentiatedTemporalModes)
            featureSpatialModes.append(spatialModes)
            featureScalars.append(scalar)

        featureTemporalModes = np.array(featureTemporalModes)
        featureDifferentiatedTemporalModes = np.array(featureDifferentiatedTemporalModes)
        featureSpatialModes = np.array(featureSpatialModes)
        featureScalars = np.array(featureScalars)

        if saveSimulation:

            filepath2 = filepath + str(numModes) + "/"
            if not os.path.exists(filepath2):
                os.makedirs(filepath2)

            np.save(filepath + features_filename, features)
            np.save(filepath2 + featureTemporalModes_filename, featureTemporalModes)
            np.save(filepath2 + featureDifferentiatedTemporalModes_filename, featureDifferentiatedTemporalModes)
            np.save(filepath2 + featureSpatialModes_filename, featureSpatialModes)
            np.save(filepath2 + featureScalars_filename, featureScalars)

            debugLogger.debug("\n\nsaved precomputed data to " + filepath + "\n")

    featureInputTrainingTemporalModes, featureOutputTrainingTemporalModes = createTrainingData(
        featureDifferentiatedTemporalModes,
        trainingLength,
        numModes,
        lookback
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        featurePredictedDifferentiatedTemporalModes = trainModel(
            simulation          = simulation,
            inputTraining       = featureInputTrainingTemporalModes,
            outputTraining      = featureOutputTrainingTemporalModes,
            predictionLength    = predictionLength,
            truncationLength    = truncationLength,
            numModes            = numModes,
            resolution          = resolution,
            numFeatures         = numFeatures,
            lookback            = lookback,
            epochs              = epochs,
            train               = train,
            load                = load,
            save                = save
        )

    featureCorrectTemporalModes = []
    featureCombinedTemporalModes = []

    for f in range(numFeatures):
        featureCorrectTemporalModes.append(getCorrectTemporalModes(features[f], featureSpatialModes[f]))
        featurePredictedDifferentiatedTemporalModes[f] = scaleData(featurePredictedDifferentiatedTemporalModes[f],
                                                                   featureScalars[f], True)
        featureCombinedTemporalModes.append(
            reconstructDifferentiation(featureTemporalModes[f], featurePredictedDifferentiatedTemporalModes[f]))
    featureCorrectTemporalModes = np.array(featureCorrectTemporalModes)

    featureCorrectDifferentiatedTemporalModes = []
    featurePredictedTemporalModes = []
    for f in range(numFeatures):
        correctDifferentiatedTemporalModes = []
        predictedTemporalModes = []
        for m in range(numModes):
            correctDifferentiatedTemporalModes.append(featureCorrectTemporalModes[f][m][trainingLength:])
            predictedTemporalModes.append(featureCombinedTemporalModes[f][m][trainingLength:])
        correctDifferentiatedTemporalModes = differentiateModes(correctDifferentiatedTemporalModes)
        featureCorrectDifferentiatedTemporalModes.append(correctDifferentiatedTemporalModes)
        featurePredictedTemporalModes.append(predictedTemporalModes)
    featureCorrectDifferentiatedTemporalModes = np.array(featureCorrectDifferentiatedTemporalModes)
    featurePredictedTemporalModes = np.array(featurePredictedTemporalModes)

    featureControlTemporalModes1 = generateControlData1(featureTemporalModes, predictionLength, numModes, numFeatures)
    featureControlTemporalModes2 = generateControlData2(featureTemporalModes, predictionLength, numModes, numFeatures)
    controlTemporalModeSets = [featureControlTemporalModes1, featureControlTemporalModes2]

    for f in range(numFeatures):
        """plotTemporalModes(
            featurePredictedDifferentiatedTemporalModes[f],
            featureCorrectDifferentiatedTemporalModes[f],
            "differentiated " + featureNames[f] + " modes"
        )"""

        plotTemporalModes(
            featurePredictedTemporalModes[f],
            featureCorrectTemporalModes[f],
            [controlData[f] for controlData in controlTemporalModeSets],
            "total " + featureNames[f] + " modes",
            8,
            trainingLength,
            truncationLength
        )

    featurePredictedData = []
    totalFeatureTemporalModes = np.array(featureCombinedTemporalModes)
    featureCorrectData = features
    featureControlDataSets = np.zeros((len(controlTemporalModeSets), numFeatures, snapshots, resolution, resolution))
    for f in range(numFeatures):
        predictedData = reconstructData(
            featureSpatialModes[f],
            totalFeatureTemporalModes[f],
            resolution,
            featureScalars[f]
        )
        featurePredictedData.append(predictedData)

        for i in range(len(controlTemporalModeSets)):
            featureControlDataSets[i][f] = reconstructData(
                featureSpatialModes[f],
                np.concatenate((featureTemporalModes[f], controlTemporalModeSets[i][f]), axis=1),
                resolution,
                featureScalars[f]
            )


    delimiterLogger.debug("number of snapshots: " + str(snapshots))
    delimiterLogger.debug("length of prediction: " + str(predictionLength))

    # featureErrors = calculateMeanSquaredError(featurePredictedData, featureCorrectData, resolution, numFeatures)
    # plotData(featureErrors, "mse over time", "snapshot", "mse", featureNames)

    plotCrossSection(featurePredictedData[0], featureCorrectData[0],
                     [featureControlData[0] for featureControlData in featureControlDataSets],
                     trainingLength + predictionLength // 2 - 1)

    plotInterval = int(time * 2)
    # plotData(featureErrors, "feature errors", "t", "MSE", featureNames)

    if plot:
        plotFlow(
            featurePredictedData,
            featureCorrectData,
            featureControlDataSets,
            featureNames,
            numFeatures,
            trainingLength,
            plotInterval,
            resolution
        )

        animatePlots(resolution, timestep)


if __name__ == "__main__":
    saveFile = "sim.txt"

    os.system("clear")

    res = 64

    main(
        train               = True,
        load                = False,
        save                = False,
        simulation          = "kelvinHelmholtz", # "vortexMerger",
        saveSimulation      = False,
        loadSimulation      = True,
        epochs              = 128,
        resolution          = res,
        numModes            = 16, #int(resolution / 4),
        time                = 2,
        timestep            = 0.001,
        plot                = True,
    )
