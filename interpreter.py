import matplotlib.pyplot as plt
import numpy as np
from lime import explanation
from lime import lime_base
import math
from keras.models import load_model
import pandas as pd

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


if __name__ == "__main__":

    data = pd.read_csv('data/global_test.csv', sep=',').astype(float)

    global_Y = data.iloc[:, -1]  # Label
    global_X = data.iloc[:, 0:-1]

    global_X_shaped = np.expand_dims(
        global_X, axis=2)

    model = getModel(CNN_MODEL_DIRECTORY)   # Load Model

    # ----------------------- Check for correct predictions ------------------------
    prediction = checkPrediction(model, global_X_shaped, global_Y)
    correct_predictions = []

    for index, value in enumerate(global_Y):
        if value == prediction[index] and value == 1:
            correct_predictions.append(index)
    # -------------------------------------------------------------------------------

    # evaluation(model, global_X_shaped, global_Y)  # Evaluate Model

    idx = 5
    num_features = 10
    num_slices = 24
    series = global_X_shaped[correct_predictions[0]]

    explainer = LimeTimeSeriesExplanation(
        class_names=['0', '1'], feature_selection='auto')
    exp = explainer.explain_instance(series, model.predict, num_features=num_features, num_samples=50, num_slices=num_slices,
                                     replacement_method='total_mean', training_set=global_X_shaped)
    exp.as_list()

    values_per_slice = math.ceil(len(series) / num_slices)
    plt.plot(series, color='b', label='Explained instance')
    plt.plot(global_X.iloc[15:, :].mean(),
             color='green', label='Mean of other class')
    plt.legend(loc='lower left')
    for i in range(num_features):
        feature, weight = exp.as_list()[i]
        start = feature * values_per_slice
        end = start + values_per_slice
        plt.axvspan(start, end, color='red', alpha=abs(weight*2))
    plt.show()
