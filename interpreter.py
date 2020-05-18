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
    y_pred = (y_pred > 0.5)
    print(np.array(
        list(zip(1-y_pred.reshape(data.shape[0]), y_pred.reshape(data.shape[0])))))
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

    # -------------------------------------------------------------------------------

    evaluation(CNN_MODEL_DIRECTORY, test_X_shaped, test_Y)  # Evaluate Model
    '''
    idx = correct_predictions[4]
    num_features = 10
    num_slices = 24
    series = test_X_shaped[idx]
    #series = test_X[0]
    global_X = test_X_shaped

    explainer = LimeTimeSeriesExplanation(
        class_names=['0', '1'], feature_selection='auto')
    exp = explainer.explain_instance(series, new_predict, num_features=num_features, num_samples=5000, num_slices=num_slices,
                                     replacement_method='total_mean', training_set=train_X_shaped, top_labels=1)
    # print(exp.available_labels()[0])
    # print(exp.as_list(label=1))
    
    explainer = LimeTimeSeriesExplanation(
        class_names=['0', '1'], feature_selection='auto')
    exp = explainer.explain_instance(series, model.predict_proba, num_features=num_features, num_samples=50, num_slices=num_slices,
                                     replacement_method='total_mean', training_set=train_X)
    # print(exp.available_labels()[0])
    print(exp.as_list())
    
    values_per_slice = math.ceil(len(series) / num_slices)
    plt.plot(series, color='b', label='Explained instance')
    plt.plot(test_X_shaped[5:, :].mean(axis=0),
             color='green', label='Mean of other class')
    plt.legend(loc='lower left')
    for i in range(num_features):
        feature, weight = exp.as_list()[i]
        start = feature * values_per_slice
        end = start + values_per_slice
        plt.axvspan(start, end, color='red', alpha=abs(weight*2))
    plt.show()
    '''
