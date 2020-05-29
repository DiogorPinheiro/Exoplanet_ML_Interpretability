import matplotlib.pyplot as plt
import numpy as np
from lime import explanation
from lime import lime_base
import math
from keras.models import load_model
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier as KNN

from lime_TimeSeries import LimeTimeSeriesExplanation
from utilities import recall_m, f1_m, precision_m, auc_roc, checkPrediction
from signalTransformer import shiftSegments, removeSegments, createGroups, cloneSegment
from evaluation import evaluation

CNN_MODEL_DIRECTORY = 'model/CNN.h5'


def getModel(model_name):
    dependencies = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m,
        'auc_roc': auc_roc
    }
    return load_model(model_name, custom_objects=dependencies)


model = getModel(CNN_MODEL_DIRECTORY)
idx = 0
global_X = []


def new_predict(data):
    y_pred = model.predict(data).reshape(-1, 1)
    y_pred = (y_pred > 0.5),
    print(y_pred)
    # print(np.array(
    #    list(zip(1-y_pred.reshape(data.shape[0]), y_pred.reshape(data.shape[0])))))
    return np.hstack((1-y_pred, y_pred))


if __name__ == "__main__":

   # ---------------------- Train Data -----------------------------------------
    data_train = np.loadtxt(
        'data/global_train.csv', delimiter=',')

    train_Y = data_train[:, -1]  # Label
    train_X = data_train[:, 0:-1]

    train_X_shaped = np.expand_dims(
        train_X, axis=2)
    # ---------------------- Test Data --------------------------------------------
    data_test = np.loadtxt(
        'data/global_test.csv', delimiter=',')

    test_Y = data_test[:, -1]  # Label
    test_X = data_test[:, 0:-1]

    test_X_shaped = np.expand_dims(
        test_X, axis=2)

    model = getModel(CNN_MODEL_DIRECTORY)   # Load Model
    # model.summary()

    # ----------------------- Check for correct predictions ------------------------

    prediction = checkPrediction(model, test_X_shaped, test_Y)
    correct_predictions = []

    for index, value in enumerate(test_Y):
        if value == prediction[index] and value == 1:
            correct_predictions.append(index)

    print(correct_predictions)
    # -------------------------------------------------------------------------------

    # evaluation(CNN_MODEL_DIRECTORY, test_X_shaped, test_Y)  # Evaluate Model

    # PC : 1, 4, 8, 9, 11, 13, 14, 16, 23, 27            |  !PC: 0, 2, 3, 5, 6, 7, 10, 12, 15, 17

    '''
    #idx = correct_predictions[0]
    idx = 2
    #groups = createGroups(test_X_shaped[idx], 40, 5)
    series = test_X_shaped[idx]
    #series = cloneSegment(2, 4, groups, test_X_shaped, idx)
    #series = shiftSegments(2, 4, groups, test_X_shaped, idx)

    explainer = LimeTimeSeriesExplanation(
        class_names=['0', '1'], feature_selection='auto')
    exp = explainer.explain_instance(series, new_predict, num_features=5, num_samples=5000, num_slices=40,
                                     replacement_method='total_mean', training_set=train_X_shaped, labels=(0, 1))

    values_per_slice = math.ceil(len(series) / 40)
    plt.plot(series, color='b', label='Explained instance')
    plt.plot(test_X_shaped[(idx+1):, :].mean(axis=0),
             color='green', label='Mean of other class')
    plt.legend(loc='lower left')
    for i in range(5):
        feature, weight = exp.as_list(label=1)[0]
        start = feature * values_per_slice
        end = start + values_per_slice
        plt.axvspan(start, end, color='red', alpha=abs(weight*2))
        plt.savefig('pos.png')
    '''

    num_features_set = [5, 10, 15, 20]
    num_slices_set = [10, 20, 30, 40]
    correct_predictions = [0, 4, 8, 11, 98, 106, 132, 164, 235, 477, 2242]
    '''
    for f in range(len(test_X_shaped)):
        plt.plot(test_X_shaped[f],
                 color='b', label='Explained instance')
        if test_Y[f] == 1:
            plt.savefig('Images/Light_Curves/Ones_'+str(f)+'.png')
        else:
            plt.savefig('Images/Light_Curves/Zeros_'+str(f)+'.png')
        plt.close()
    '''

    for slices in num_slices_set:

        for feat in num_features_set:
            num_features = feat
            num_slices = slices

            for i in range(10):     # Positives
                idx = correct_predictions[i]
                series = test_X_shaped[idx]

                explainer = LimeTimeSeriesExplanation(
                    class_names=['0', '1'], feature_selection='auto')
                exp = explainer.explain_instance(series, new_predict, num_features=num_features, num_samples=5000, num_slices=num_slices,
                                                 replacement_method='total_mean', training_set=train_X_shaped, labels=(0, 1))
                # print(exp.available_labels()[0])
                # print(exp.as_list(label=1))

                values_per_slice = math.ceil(len(series) / num_slices)
                plt.plot(series, color='b', label='Explained instance')
                plt.plot(test_X_shaped[(idx+1):, :].mean(axis=0),
                         color='green', label='Mean of other class')
                plt.legend(loc='lower left')
                for i in range(num_features):
                    feature, weight = exp.as_list()[i]
                    start = feature * values_per_slice
                    end = start + values_per_slice
                    plt.axvspan(start, end, color='red', alpha=abs(weight*2))
                plt.savefig("slices_"+str(slices)+"feat_" +
                            str(feat)+"/"+str(idx)+'_positive.png')
                plt.close()

            negatives = [0, 2, 3, 5, 6, 7, 10, 12, 15, 17]
            for i in negatives:     # Negatives
                series = test_X_shaped[i]

                explainer = LimeTimeSeriesExplanation(
                    class_names=['0', '1'], feature_selection='auto')
                exp = explainer.explain_instance(series, new_predict, num_features=num_features, num_samples=5000, num_slices=num_slices,
                                                 replacement_method='total_mean', training_set=train_X_shaped, labels=(0, 1))
                # print(exp.available_labels()[0])
                # print(exp.as_list(label=1))

                values_per_slice = math.ceil(len(series) / num_slices)
                plt.plot(series, color='b', label='Explained instance')
                plt.plot(test_X_shaped[(i+1):, :].mean(axis=0),
                         color='green', label='Mean of other class')
                plt.legend(loc='lower left')
                for i in range(num_features):
                    feature, weight = exp.as_list(label=0)[i]
                    start = feature * values_per_slice
                    end = start + values_per_slice
                    plt.axvspan(start, end, color='red', alpha=abs(weight*2))
                plt.savefig("slices_"+str(slices)+"feat_" +
                            str(feat)+"/"+str(idx)+'_negative.png')
                plt.close()
