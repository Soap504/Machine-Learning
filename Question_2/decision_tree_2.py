# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and evaluate decision tree models on different datasets
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard dictionaries, lists, and arrays.

# Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    feature_mapping = {
        'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3,
        'Myope': 1, 'Hypermetrope': 2,
        'No': 1, 'Yes': 2,
        'Reduced': 1, 'Normal': 2
    }
    for row in dbTraining:
        X.append([feature_mapping[row[0]], feature_mapping[row[1]], feature_mapping[row[2]], feature_mapping[row[3]]])
        Y.append(feature_mapping[row[4]])

    # Loop your training and test tasks 10 times here
    accuracies = []
    for i in range(10):
        # Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        correct_predictions = 0
        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            test_instance = [feature_mapping[data[0]], feature_mapping[data[1]], feature_mapping[data[2]],
                             feature_mapping[data[3]]]

            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            class_predicted = clf.predict([test_instance])[0]

            # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_predicted == feature_mapping[data[4]]:
                correct_predictions += 1

        # Compute accuracy
        accuracy = correct_predictions / len(dbTest)
        accuracies.append(accuracy)

    # Find the average of this model during the 10 runs (training and test set)
    avg_accuracy = sum(accuracies) / len(accuracies)

    # Print the average accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {avg_accuracy:.4f}")
