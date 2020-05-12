import matplotlib.pyplot as plt
import numpy as np
import shap
import math
from keras.models import load_model
import pandas as pd

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

    #random_ind = np.random.choice(
    #    global_X_shaped.shape[0], 1000, replace=False)
    #data = global_X_shaped[random_ind[0:50]]
    #e = shap.DeepExplainer(
    #    (model.layers[0].input, model.layers[-1].output), data)
    #values = e.shap_values(global_X_shaped[:500])
    #shap.summary_plot(values[0], global_X_shaped[:500])
    #shap.force_plot(e.expected_value,values[0])

    #model_summary = shap.kmeans(global_X_shaped,25)
    explainer = shap.DeepExplainer(model,global_X_shaped[:50])
    shap_values =  explainer.shap_values(global_X_shaped[:50])
    shap.summary_plot(shap_values[0][0],global_X_shaped[0])
