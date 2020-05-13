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

    # ---------------------- Train Data -----------------------------------------
    data_train = pd.read_csv('data/global_train.csv', sep=',').astype(float)

    train_Y = data_train.iloc[:, -1]  # Label
    train_X = data_train.iloc[:, 0:-1]

    train_X_shaped = np.expand_dims(
        train_X, axis=2)
    # ---------------------- Test Data --------------------------------------------
    data_test = pd.read_csv('data/global_test.csv', sep=',').astype(float)

    test_Y = data_test.iloc[:, -1]  # Label
    test_X = data_test.iloc[:, 0:-1]

    test_X_shaped = np.expand_dims(
        test_X, axis=2)

    model = getModel(CNN_MODEL_DIRECTORY)   # Load Model

    # ----------------------- Check for correct predictions ------------------------
    prediction = checkPrediction(model, test_X_shaped, test_Y)
    correct_predictions = []

    for index, value in enumerate(test_Y):
        if value == prediction[index] and value == 1:
            correct_predictions.append(index)
    # -------------------------------------------------------------------------------

    # evaluation(model, test_X_shaped, test_Y)  # Evaluate Model

    # random_ind = np.random.choice(
    #    test_X_shaped.shape[0], 1000, replace=False)
    # data = test_X_shaped[random_ind[0:50]]
    # e = shap.DeepExplainer(
    #    (model.layers[0].input, model.layers[-1].output), data)
    # values = e.shap_values(test_X_shaped[:500])
    # shap.summary_plot(values[0], test_X_shaped[:500])
    # shap.force_plot(e.expected_value,values[0])

    # model_summary = shap.kmeans(test_X_shaped,25)
    explainer = shap.DeepExplainer(model, train_X_shaped[:50])
    shap_values = explainer.shap_values(test_X_shaped[:1])
    y_pred = model.predict(test_X_shaped[:1])
    print('Actual Category: %s, Predict Category: %s' % (test_Y[0], y_pred[0]))
    shap.force_plot(explainer.expected_value[0], shap_values[0][0]
    # shap.summary_plot(shap_values[0][0], test_X_shaped[0])
