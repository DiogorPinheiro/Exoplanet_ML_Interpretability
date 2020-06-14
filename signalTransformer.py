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

        @param mean_value (float): mean value of data
        @param array (list[list[float]]): light curve data

        @return [list[float]): created array

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

        @param data (np.ndarray)

        @return (list[float])

    '''
    aux = []
    for v1, x in enumerate(data):
        for v2, y in enumerate(x):
            for v3, z in enumerate(y):
                aux.append(data[v1][v2][v3])
    return aux


def replace_curve(indexes, data, mean_value):
    '''
        Replace slices of the light curve with its mean value.

        @param indexes (list[int]): indexes where the change will be made
        @param data (np.ndarray)
        @param mean_value (float): mean value of the data

        @return (list[list[float]])

    '''
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
    '''
        Change segment from index1 to index2 and vice-versa.

        @param index1 (int): first index to change data
        @param index2 (int): second index to change data
        @param data (np.ndarray)

        @return (list[list[float]])

    '''
    aux = data[index1]
    data[index1] = data[index2]
    data[index2] = aux
    return data


def clone_signal(index1, index2, data):
    '''
        Replace segment at index2 with the segment at index1.

        @param index1 (int): first index to change data
        @param index2 (int): second index to change data
        @param data (np.ndarray)

        @return (list[list[float]])

    '''
    data[index2] = data[index1]
    return data

# ----------------------------- Main Functions --------------------------------


def createGroups(data, num_chunks=40, divider_size=5):
    '''
        Slice data into chunks and group them.

        @param data (np.ndarray)
        @param num_chunks (int): Number of chunks the data will be "sliced"
        @param divider_size (int): number of groups

        @return (np.ndarray)

    '''

    chunks = np.array_split(data, num_chunks)
    groups = np.array_split(chunks, divider_size)

    return groups


def removeSegments(indexes, groups, data, posIndex):
    '''
        Replaces certain groups with the mean value of the data.

        @param indexes (list[int]): indexes of the groups that will be changed
        @param groups (np.ndarray): data divided in groups
        @param data (np.ndarray)
        @param posIndex (int): row of the light curve to be modified 

        @return (np.ndarray)

    '''

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
    '''
        Shift two groups (index 1 goes to index 2 and vice-versa).

        @param index1 (int): first index to change data
        @param index2 (int): second index to change data
        @param groups (np.ndarray): data divided in groups
        @param data (np.ndarray)
        @param posIndex (int): row of the light curve to be modified 

        @return (np.ndarray)

    '''

    shifted_data = shift_signal(index1, index2, groups)
    new_curve = groupToPoints(shifted_data)
    new_curve = np.array(new_curve)

    data_copy = data.copy()
    data_copy[posindex] = list(
        new_curve.reshape((new_curve.shape[0], 1)))

    #chunkVisualization(data[posindex], 400)

    return data_copy[posindex]


def cloneSegment(index1, index2, groups, data, posindex):
    '''
        Clone two groups (index 1 goes to index 2 and vice-versa).

        @param index1 (int): first index to change data
        @param index2 (int): second index to change data
        @param groups (np.ndarray): data divided in groups
        @param data (np.ndarray)
        @param posIndex (int): row of the light curve to be modified 

        @return (np.ndarray)

    '''
    cloned_data = clone_signal(index1, index2, groups)
    new_curve = groupToPoints(cloned_data)
    new_curve = np.array(new_curve)

    data_copy = data.copy()
    data_copy[posindex] = list(
        new_curve.reshape((new_curve.shape[0], 1)))

    #chunkVisualization(data[posindex], 400)

    return data_copy[posindex]
