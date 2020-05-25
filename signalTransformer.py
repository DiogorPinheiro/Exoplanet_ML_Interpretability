import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import shuffle
from collections import Counter

from utilities import chunkVisualization


def buildArray(mean_value, array):
    '''
        Builds an array with the light curve mean value

        input: mean_value (float)
               array (list of lists )
        output: array of floats
    '''
    aux = []
    for index, i in enumerate(array):
        for index2, i2 in enumerate(i):
            aux.append(mean_value)
    # Use numpy to reshape array
    aux = np.array(aux)
    aux = aux.reshape((aux.shape[0], 1))

    return list(aux)


def groupToPoints(data):
    '''
        Convert light curve divided by groups to a list of points

        input: data (list of lists of lists)
        output: list
    '''
    aux = []
    for v1, x in enumerate(data):
        for v2, y in enumerate(x):
            for v3, z in enumerate(y):
                aux.append(data[v1][v2][v3])
    return aux


def replace_curve(indexes, data, mean_value):
    aux = []

    for i, d in enumerate(data):
        if i not in indexes:
            arr = buildArray(mean_value, d)
            # Get array with the mean value
            aux.append(arr)
        else:
            aux.append(d)
    return aux


def shift_signal(index1, index2, data):
    aux = data[index1]
    data[index1] = data[index2]
    data[index2] = aux
    return data

# ----------------------------- Main Functions --------------------------------


def createGroups(data, num_chunks=40, divider_size=5):
    chunks = np.array_split(data, num_chunks)
    groups = np.array_split(chunks, divider_size)

    return groups


def removeSegments(indexes, groups, data, posIndex):
    mean_value = np.mean(data)   # Light Curve Mean Value
    replace_curve(indexes, groups, mean_value)

    new_curve = groupToPoints(data)
    new_curve = np.array(new_curve)

    data_copy = data.copy()
    data_copy[posIndex] = list(
        new_curve.reshape((new_curve.shape[0], 1)))

    #chunkVisualization(global_X_copy[1], 400)

    return data_copy[posIndex]


def shiftSegments(index1, index2, groups, data, posindex):
    shifted_data = shift_signal(1, 2, groups)
    new_curve = groupToPoints(shifted_data)
    new_curve = np.array(new_curve)

    data_copy = data.copy()
    data_copy[posindex] = list(
        new_curve.reshape((new_curve.shape[0], 1)))

    #chunkVisualization(data[posindex], 400)

    return data_copy[posindex]

# Clonar Pico
