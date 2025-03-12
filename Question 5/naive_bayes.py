#-------------------------------------------------------------------------
# AUTHOR: Ali Khaja
# FILENAME: naive_bayes.py
# SPECIFICATION: A Naive Bayes classifier implementation to classify weather instances
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hrs
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# Reading the training data in a csv file
training_data = []
with open('weather_training.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        training_data.append(row)

# Transform the original training features to numbers and add them to the 2D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
weather_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2, 'Low': 3}
wind_map = {'Weak': 1, 'Strong': 2}

X = []
Y = []

# Skip the header and process the data, excluding the Day column
for row in training_data[1:]:  # Skip header row
    weather = weather_map.get(row[1], 0)  # Get the weather (Outlook) value, default to 0 if invalid
    temp = temp_map.get(row[2], 0)        # Get the temperature value, default to 0 if invalid
    humidity = humidity_map.get(row[3], 0)  # Get the humidity value, default to 0 if invalid
    wind = wind_map.get(row[4], 0)        # Get the wind value, default to 0 if invalid
    classification = 1 if row[5] == 'Yes' else 2  # Yes = 1, No = 2
    
    # Add processed data to X and Y
    X.append([weather, temp, humidity, wind])
    Y.append(classification)

# Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

# Reading the test data in a csv file
test_data = []
with open('weather_test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_data.append(row)

# Printing the header of the solution
print("Outlook, Temperature, Humidity, Wind, Predicted Class, Confidence")

# Use your test samples to make probabilistic predictions.
for row in test_data[1:]:  # Skip header row
    weather = weather_map.get(row[1], 0)  # Get the weather (Outlook) value
    temp = temp_map.get(row[2], 0)        # Get the temperature value
    humidity = humidity_map.get(row[3], 0)  # Get the humidity value
    wind = wind_map.get(row[4], 0)        # Get the wind value
    
    # Make the prediction
    probs = clf.predict_proba([[weather, temp, humidity, wind]])[0]
    
    # Check if the classification confidence is >= 0.75
    if max(probs) >= 0.75:
        predicted_class = 'Yes' if probs[0] > probs[1] else 'No'
        confidence = max(probs)
        print(f"{row[1]}, {row[2]}, {row[3]}, {row[4]}, {predicted_class}, {confidence:.2f}")
