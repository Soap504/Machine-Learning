#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard dictionaries, lists, and arrays.

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
X = []
Y = []
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}
play_map = {'Yes': 1, 'No': 2}

with open("weather_training.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        X.append([outlook_map[row[1]], temp_map[row[2]], humidity_map[row[3]], wind_map[row[4]]])
        Y.append(play_map[row[5]])

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# (Already done above)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# (Already done above)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
with open("weather_test.csv", "r") as file:
    reader = csv.reader(file)
    test_data = [row for row in reader][1:]  # Skip header and read data

#Printing the header of the solution
#--> add your Python code here
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in test_data:
    instance = [outlook_map[row[1]], temp_map[row[2]], humidity_map[row[3]], wind_map[row[4]]]
    probabilities = clf.predict_proba([instance])[0]
    prediction = clf.predict([instance])[0]
    confidence = max(probabilities)

    if confidence >= 0.75:
        predicted_label = "Yes" if prediction == 1 else "No"
        print(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {predicted_label} {confidence:.3f}")
