#-------------------------------------------------------------------------
# AUTHOR: Ali Khaja
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and test decision tree models on different datasets and then get the average accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. 
# You have to work here only with standard dictionaries, lists, and arrays.

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Mapping categorical values to numbers
feature_mapping = {
    "Young": 1, "Prepresbyopic": 2, "Presbyopic": 3,
    "Myope": 1, "Hypermetrope": 2,
    "Yes": 1, "No": 2,
    "Reduced": 1, "Normal": 2
}

class_mapping = {"Yes": 1, "No": 2}

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
    for row in dbTraining:
        X.append([feature_mapping[row[0]], feature_mapping[row[1]], feature_mapping[row[2]], feature_mapping[row[3]]])

    # Transform the original categorical training classes to numbers and add to the vector Y.
    for row in dbTraining:
        Y.append(class_mapping[row[4]])

    accuracy_sum = 0  # To accumulate accuracy over 10 runs

    # Loop your training and test tasks 10 times here
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
        total_predictions = len(dbTest)

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training
            test_instance = [feature_mapping[data[0]], feature_mapping[data[1]], feature_mapping[data[2]], feature_mapping[data[3]]]

            # Use the decision tree to make the class prediction
            class_predicted = clf.predict([test_instance])[0]

            # Compare the prediction with the true label (located at data[4]) to calculate accuracy
            if class_predicted == class_mapping[data[4]]:
                correct_predictions += 1

        # Calculate accuracy for this run
        accuracy = correct_predictions / total_predictions
        accuracy_sum += accuracy

    # Find the average accuracy of this model during the 10 runs (training and test set)
    final_accuracy = accuracy_sum / 10

    # Print the average accuracy of this model during the 10 runs
    print(f"Final accuracy when training on {ds}: {final_accuracy:.2f}")
