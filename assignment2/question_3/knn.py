# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: knn.py
# SPECIFICATION: Compute the LOO-CV error rate for a 1NN classifier on spam/ham classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

# File path to the dataset
#csv_path = "C:\\Users\\sophi\\OneDrive\\Desktop\\Python\\MachineLearning\\Assignment2\\Question_3\\CSV_Files\\email_classification.csv"

db = []

# Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

error_count = 0
n = len(db)

# Loop your data to allow each instance to be your test set
for i in range(n):

    X = []  # Training features
    Y = []  # Training labels

    for j in range(n):
        if i != j:  # Exclude the instance that will be used for testing
            X.append([float(value) for value in db[j][:-1]])  # Convert features to float
            Y.append(1 if db[j][-1] == 'spam' else 0)  # Convert class labels to 1 (spam) and 0 (ham)

    # Store the test sample of this iteration in the vector testSample
    testSample = [float(value) for value in db[i][:-1]]
    true_label = 1 if db[i][-1] == 'spam' else 0

    # Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # Use your test sample in this iteration to make the class prediction.
    class_predicted = clf.predict([testSample])[0]

    # Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label:
        error_count += 1

# Print the error rate
error_rate = error_count / n
print(f"LOO-CV Error Rate: {error_rate:.4f}")
