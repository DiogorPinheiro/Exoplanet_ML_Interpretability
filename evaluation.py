import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

from utilities import auc_roc, f1_m, precision_m, recall_m


def evaluate(model_name, data_X, data_y):
    # Model Dependencies
    dependencies = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m,
        'auc_roc': auc_roc,
        'num_classes': 1,
        'input_shape': (data_X.shape[1], 1)
    }

    # Get Model
    model = keras.models.load_model(model_name, custom_objects=dependencies)

    score_loss = []
    score_acc = []
    score_f1 = []
    score_prec = []
    score_rec = []
    score_auc = []
    for i in range(50):
        data_X_shuf, data_y_shuf = shuffle(data_X, data_y)

        # Evaluate Model
        score = model.evaluate(data_X_shuf, data_y_shuf, verbose=0)

        score_loss.append(score[0])
        score_acc.append(score[1])
        score_f1.append(score[2])
        score_prec.append(score[3])
        score_rec.append(score[4])
        score_auc.append(score[5])

        # Print results
        print("\n------------------ Model : {} ---------------------".format(model_name))
        print("{}: {:0.2f} ".format(
            model.metrics_names[0], np.mean(score_loss)))
        print("{}: {:0.2f} ".format(
            model.metrics_names[1], np.mean(score_acc)))
        print("{}: {:0.2f} ".format(model.metrics_names[2], np.mean(score_f1)))
        print("{}: {:0.2f} ".format(
            model.metrics_names[3], np.mean(score_prec)))
        print("{}: {:0.2f} ".format(
            model.metrics_names[4], np.mean(score_rec)))
        print("{}: {:0.2f} ".format(
            model.metrics_names[5], np.mean(score_auc)))
        print("---------------------------------------------------")


def evaluation(model, global_X, global_Y):

    #global_X = np.expand_dims(global_X, axis=2)

    evaluate(model, global_X, global_Y)
