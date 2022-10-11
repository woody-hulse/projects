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
from keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import numpy as np
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
    :param data:        input data with shape (features, snapshots, x, y)
    :return:            output data with shape (snapshots, x, y, features)
    """

    newData = []

    for s in range(len(data[0])):
        snapshotData = []
        for x in range(len(data[0][0])):
            rowData = []
            for y in range(len(data[0][0][0])):
                pointData = []
                for f in range(len(data)):
                    pointData.append(data[f][s][x][y])
                rowData.append(pointData)
            snapshotData.append(rowData)
        newData.append(snapshotData)

    newData = np.asarray(newData)

    return newData


def unpackData(data):

    """
    :param data:        input data with shape (snapshots, x, y, features)
    :return:            output data with shape (features, snapshots, x, y)
    """

    newData = []

    for f in range(len(data[0][0][0])):
        featureData = []
        for s in range(len(data)):
            snapshotData = []
            for x in range(len(data[0])):
                rowData = []
                for y in range(len(data[0][0])):
                    rowData.append(data[s][x][y][f])
                snapshotData.append(rowData)
            featureData.append(snapshotData)
        newData.append(featureData)

    newData = np.asarray(newData)

    return newData


def createModel(xDimensions, yDimensions, numFeatures, summary=True):

    """
    create an lstm neural network (four-dimensional) to accomidate multi-featured, two-dimensional data
    :param xDimensions:         number of cells in x dimension of data
    :param yDimensions:         number of cells in y dimension of data
    :param numFeatures:         number of features [rho, x velocity, y velocity, pressure] in dataset
    :param summary:             boolean to print model information
    :return:
    """

    shape = (xDimensions, yDimensions, numFeatures)

    inputLayer = layers.Input(shape=shape)


    """
    
    LSTMLayer = layers.TimeDistributed(layers.LSTM(int(xDimensions / 2), return_sequences=True))(inputLayer)
    FlatteningLayer1 = layers.Flatten(input_shape=shape)(LSTMLayer)
    DenseLayer1 = layers.Dense(int(xDimensions / 2))(FlatteningLayer1)
    DropoutLayer1 = layers.Dropout(0.2)(DenseLayer1)
    DenseLayer2 = layers.Dense(int(xDimensions / 2))(DropoutLayer1)
    DropoutLayer2 = layers.Dropout(0.2)(DenseLayer2)
    DenseLayer3 = layers.Dense(int(xDimensions * yDimensions * numFeatures), activation='tanh')(DropoutLayer2)
    OutputLayer = layers.Reshape(shape)(DenseLayer3)


    """

    # Dense Architecture
    dLSTMLayer = layers.TimeDistributed(layers.LSTM(numFeatures, return_sequences=True))(inputLayer)
    dFlatteningLayer1 = layers.Flatten(input_shape=shape)(dLSTMLayer)
    dDenseLayer1 = layers.Dense(xDimensions * 8)(dFlatteningLayer1)
    dDropoutLayer1 = layers.Dropout(0.2)(dDenseLayer1)
    dDenseLayer2 = layers.Dense(xDimensions * 8)(dDropoutLayer1)
    dDropoutLayer2 = layers.Dropout(0.2)(dDenseLayer2)
    dDenseLayer3 = layers.Dense(16 * numFeatures, activation='tanh')(dDropoutLayer2)
    dOutputLayer = layers.Flatten()(dDenseLayer3)

    # Convolutional Architecture
    cConvolutionLayer1 = layers.Convolution2D(xDimensions, (2, 2), input_shape=shape, padding="same")(inputLayer)
    cPoolingLayer1 = layers.MaxPooling2D(pool_size=2, padding="same")(cConvolutionLayer1)
    cLSTMLayer = layers.TimeDistributed(layers.LSTM(xDimensions * 8, return_sequences=True))(cPoolingLayer1)
    cDropoutLayer1 = layers.Dropout(0.2)(cLSTMLayer)
    cConvolutionLayer2 = layers.Convolution2D(xDimensions, (2, 2), padding="same")(cDropoutLayer1)
    cPoolingLayer2 = layers.MaxPooling2D(pool_size=2, padding="same")(cConvolutionLayer2)
    cDenseLayer1 = layers.Dense(16 * numFeatures, activation='tanh')(cPoolingLayer2)
    cOutputLayer = layers.Flatten()(cDenseLayer1)

    # Combined Output
    concatenateLayer = layers.concatenate([cOutputLayer, dOutputLayer])
    combineDenseLayer = layers.Dense(xDimensions * yDimensions * numFeatures)(concatenateLayer)
    outputLayer = layers.Reshape(shape)(combineDenseLayer)


    model = keras.Model(inputLayer, outputLayer)

    model.compile(loss='mse', optimizer='adam')

    if summary:
        model.summary()

    debugLogger.debug("model created")

    return model

def trainModel(
        inputTraining,
        outputTraining,
        inputTesting,
        xDimensions,
        yDimensions,
        numFeatures,
        epochs,
        train,
        load,
        save,
        plot=True
):

    """
    loads/generates/saves and trains lstm model
    :param inputTraining:       input training dataset, shape=(features, snapshots, x, y)
    :param outputTraining:      output training dataset, shape=(features, snapshots, x, y)
    :param inputTesting:        input testing dataset, shape=(features, snapshots, x, y)
    :param xDimensions:         number of cells in x dimension of data
    :param yDimensions:         number of cells in y dimension of data
    :param numFeatures:         number of features [rho, x velocity, y velocity, pressure] in dataset
    :param epochs:              number of training rounds
    :param train:               boolean to train model before fitting
    :param load:                boolean to load model from file
    :param save:                boolean to save model to file
    :param plot:                boolean to plot the loss/epoch data
    :return:                    predicted feature dataset from inputTesting set, shape=(features, snapshots, x, y)
    """

    for snap in range(len(inputTraining[0])):
        plt.cla()
        plt.subplot(1, 2, 1)
        plt.imshow(inputTraining[0][snap].T)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        plt.subplot(1, 2, 2)
        plt.imshow(outputTraining[0][snap].T)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        plt.pause(0.0001)

    inputTraining = reorganizeData(inputTraining)
    outputTraining = reorganizeData(outputTraining)
    inputTesting = reorganizeData(inputTesting)


    model = createModel(xDimensions, yDimensions, numFeatures)

    filename = "model" + str(xDimensions)

    if load:
        model = load_model("savedModels3/" + filename)

        debugLogger.debug("model loaded from file " + filename)

    batchSize = 1

    if train:
        if save:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath="savedModels3/" + filename,
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

    predictBatchSize = 1
    numPrediction = len(inputTesting)
    predicted = inputTesting[:predictBatchSize]
    for i in range(math.floor(numPrediction / predictBatchSize)):
        predicted = np.append(
            predicted,
            model.predict(np.reshape(predicted[-1*predictBatchSize:],
                                     (predictBatchSize, xDimensions, yDimensions, numFeatures))),
            axis=0
        )

    predicted = predicted[predictBatchSize:]
    predicted = unpackData(predicted)

    debugLogger.debug("forecast data generated from model")

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


def POD(data, trainingRatio, optimalModes=28):

    """
    computes POD basis vectors and modal coefficients
    :param data:                single feature data
    :param trainingRatio:       ratio of training to testing data
    :param optimalModes:        number of optimal modal coefficients to use
    :return:                    modal coefficients and list of basis vectors
    """

    snapshots = len(data)

    reshapedData = []
    for i in range(snapshots):
        reshapedData.append(linearize(data[i]))

    u, s, v = np.linalg.svd(reshapedData, full_matrices=True)
    basisVectors = u[:, :optimalModes]

    modalCoefficients = np.matmul(np.transpose(basisVectors), reshapedData)

    modalCoefficients = [undoLinearize(listData, len(data[0])) for listData in modalCoefficients]

    debugLogger.debug("computed pod basis vectors and modal coefficients")

    return modalCoefficients, basisVectors


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


def createTrainingTestingData(modalCoefficients, snapshots, xDimensions, yDimensions, trainingRatio=0.8):

    """
    partitions data into training and testing sections at 4:1 ratio, offsets output dataset by 1
    :param modalCoefficients:   single-feature modal coefficients of data
    :param snapshots:           number of snapshots in dataset
    :param xDimensions:         number of cells in x dimension of data
    :param yDimensions:         number of cells in y dimension of data
    :param trainingRatio:       ratio of training dataset size to testing dataset size
    :return:                    input/output training/testing datasets
    """

    modalCoefficients, scalar = scaleData(modalCoefficients)

    trainLength = math.floor(snapshots * trainingRatio)
    testLength = snapshots - trainLength

    inputTrainingModalCoefficients = np.reshape(modalCoefficients[:trainLength - 1],
                                                 (trainLength - 1, xDimensions, yDimensions))
    outputTrainingModalCoefficients = np.reshape(modalCoefficients[1:trainLength],
                                                  (trainLength - 1, xDimensions, yDimensions))

    inputTestingModalCoefficients = np.reshape(modalCoefficients[trainLength - 1:snapshots - 1],
                                                (testLength, xDimensions, yDimensions))
    outputTestingModalCoefficients = np.reshape(modalCoefficients[trainLength:snapshots],
                                                 (testLength, xDimensions, yDimensions))

    debugLogger.debug("created training and testing datasets")

    return inputTrainingModalCoefficients,\
           outputTrainingModalCoefficients, \
           inputTestingModalCoefficients, \
           outputTestingModalCoefficients, \
           scalar


def reconstructData(modalCoefficients, basisVectors, trainingRatio, resolution, scalar):

    """
    reconstruct data from predicted modal coefficients
    :param modalCoefficients:           modal coefficients of data
    :param basisVectors:                basis vectors from earlier datasets
    :param trainingRatio                ratio of training to testing data
    :param resolution                   dimensions of dataset
    :param scalar:                      scalar to reverse normalization
    :return:                            reconstructed predicted data
    """

    # modalCoefficients = modalCoefficients[int(len(modalCoefficients) * (1 - trainingRatio)):]
    modalCoefficients = [linearize(d) for d in modalCoefficients]
    reconstructedData = np.matmul(basisVectors, modalCoefficients)
    reconstructedData = [undoLinearize(listData, resolution) for listData in reconstructedData]
    reconstructedData = [scaleData(data, scalar, inverse=True) for data in reconstructedData]

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

def plotFlow(featurePredictedData, featureCorrectData, featureNames, numFeatures, start, interval, resolution, vxvy=True):

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
        for j in range(start, dataLength, (dataLength - start) // 50):
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
                elif featureNames[i] == "x velocity" or featureNames[i] == "y velocity" or featureNames[i] == "velocity":
                    vectorSpacing = int(resolution / 16)
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
         numCoefficients=28,
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
    :param numCoefficients  number of basis vectors and modal coefficients
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

    featureInputTrainingModalCoefficients, \
    featureOutputTrainingModalCoefficients, \
    featureInputTestingModalCoefficients, \
    featureOutputTestingModalCoefficients = [], [], [], []

    featureBases = []
    featureScalars = []
    trainingRatio = 0.5 # keep

    featureIndex = 0
    for data in features:

        delimiterLogger.debug(featureNames[featureIndex])
        featureIndex += 1

        modalCoefficients, basisVectors = POD(data, trainingRatio, numCoefficients)

        inputTrainingModalCoefficients, \
        outputTrainingModalCoefficients, \
        inputTestingModalCoefficients, \
        outputTestingModalCoefficients, \
        scalar = createTrainingTestingData(
            modalCoefficients,
            numCoefficients,
            xDimensions,
            yDimensions,
            trainingRatio
        )

        # featureMeans.append(mean)
        featureBases.append(basisVectors)
        featureScalars.append(scalar)

        featureInputTrainingModalCoefficients.append(inputTrainingModalCoefficients)
        featureOutputTrainingModalCoefficients.append(outputTrainingModalCoefficients)
        featureInputTestingModalCoefficients.append(inputTestingModalCoefficients)
        featureOutputTestingModalCoefficients.append(outputTestingModalCoefficients)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        featurePredictedModalCoefficients = trainModel(
            np.array(featureInputTrainingModalCoefficients),
            np.array(featureOutputTrainingModalCoefficients),
            np.array(featureInputTestingModalCoefficients),
            xDimensions,
            yDimensions,
            numFeatures,
            epochs,
            train,
            load,
            save
        )

        viewCoefficients = True
        if viewCoefficients:
            for f in range(1):
                for i in range(len(featurePredictedModalCoefficients[f])):
                    plt.cla()
                    plt.subplot(1, 2, 1)
                    plt.imshow(featurePredictedModalCoefficients[f][i].T)
                    ax = plt.gca()
                    ax.invert_yaxis()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_aspect('equal')

                    plt.subplot(1, 2, 2)
                    plt.imshow(featureOutputTestingModalCoefficients[f][i].T)
                    ax = plt.gca()
                    ax.invert_yaxis()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_aspect('equal')
                    plt.pause(0.01)

            for i in range(len(featurePredictedModalCoefficients[0]) - 1):
                # print(featurePredictedModalCoefficients[0][i] == featurePredictedModalCoefficients[0][i + 1])
                pass


    featurePredictedData = []
    featureCorrectData = []
    for i in range(numFeatures):
        totalPredictedModalCoefficients = np.concatenate(
            (featureInputTrainingModalCoefficients[i],
             [featureOutputTrainingModalCoefficients[i][-1]],
             featurePredictedModalCoefficients[i]),
            axis=0
        )
        predictedData = reconstructData(
            totalPredictedModalCoefficients,
            featureBases[i],
            trainingRatio,
            resolution,
            featureScalars[i]
        )
        featurePredictedData.append(predictedData)

        totalCorrectModalCoefficients = np.concatenate(
            (featureInputTrainingModalCoefficients[i],
             [featureOutputTrainingModalCoefficients[i][-1]],
             featureOutputTestingModalCoefficients[i]),
             axis=0
        )
        correctData = reconstructData(
            totalCorrectModalCoefficients,
            featureBases[i],
            trainingRatio,
            resolution,
            featureScalars[i]
        )
        featureCorrectData.append(correctData)

    predictionLength = int(len(featurePredictedData[0]))
    delimiterLogger.debug("number of snapshots: " + str(snapshots))
    delimiterLogger.debug("length of prediction: " + str(predictionLength))

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
            [feature[snapshots - predictionLength:] for feature in features],
            featureNames,
            numFeatures,
            0,
            plotInterval,
            resolution,
            vxvy=True
        )

        animatePlots()


if __name__ == "__main__":

    saveFile = "sim.txt"

    # train, load, save, save sim, load sim, epochs, resolution, time, plot
    # savefile format is save[resolution]_[time]_[timestep].txt
    main(
        False, # Train model
        True, # Load model from file
        True, # Save model to file
        True, # Save simulation data to file
        False, # Load simulation data from file
        1000, # Training epochs
        16, # Data resolution
        28, # Number of coefficients
        2, # Time multiplier # temporarily inactive (2)
        0.01, # Timestep length
        True, # Plot data
    )