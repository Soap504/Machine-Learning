#-------------------------------------------------------------------------
# AUTHOR: Sophia Arce
# FILENAME: perceptron.py
# SPECIFICATION: Train Perceptron and MLP classifiers on handwritten digit data with varying hyperparameters.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1.5 hours
#-------------------------------------------------------------------------
# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # learning rates
r = [True, False]  # shuffle options

df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the data by using Pandas library

X_training = np.array(df.values)[:, :-1]  # updated slicing for clarity
y_training = np.array(df.values)[:, -1]

df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the data by using Pandas library

X_test = np.array(df.values)[:, :-1]  # updated slicing for clarity
y_test = np.array(df.values)[:, -1]

highest_perceptron_accuracy = 0.0
highest_mlp_accuracy = 0.0

for rate in n:  # iterates over learning rates
    for shuffle in r:  # iterates over shuffle options

        # --> Updated Python code
        for algorithm in ['Perceptron', 'MLP']:  # iterates over the algorithms

            # Create a Neural Network classifier
            if algorithm == 'Perceptron':
                clf = Perceptron(eta0=rate, shuffle=shuffle, max_iter=1000, tol=1e-3)
            else:
                clf = MLPClassifier(
                    activation='logistic',
                    learning_rate_init=rate,
                    hidden_layer_sizes=(25,),
                    shuffle=shuffle,
                    max_iter=1000
                )

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # make the classifier prediction for each test sample and start computing its accuracy
            correct = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]  # extracting scalar with [0]
                if prediction == y_testSample:
                    correct += 1
            accuracy = correct / len(y_test)

            # check if the calculated accuracy is higher than the previously one calculated for each classifier
            if algorithm == 'Perceptron' and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(
                    f"Highest Perceptron accuracy so far: {accuracy:.4f}, Parameters: learning rate={rate}, shuffle={shuffle}")
            elif algorithm == 'MLP' and accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy = accuracy
                print(
                    f"Highest MLP accuracy so far: {accuracy:.4f}, Parameters: learning rate={rate}, shuffle={shuffle}")