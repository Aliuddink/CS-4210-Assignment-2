#-------------------------------------------------------------------------
# AUTHOR: Ali Khaja
# FILENAME: knn.py
# SPECIFICATION Leave-One-Out Cross-Validation (LOO-CV) error rate for 1NN.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

# Initialize error counter
error_count = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):
    
    X = []  # Training features
    Y = []  # Training labels
    
    for j in range(len(db)):
        if i != j:  # Exclude the test instance
            X.append([float(x) for x in db[j][:-1]])  # Convert features to float
            Y.append(1 if db[j][-1] == 'spam' else 0)  # Convert labels to numeric
    
    #Store the test sample of this iteration in the vector testSample
    testSample = [float(x) for x in db[i][:-1]]
    true_label = 1 if db[i][-1] == 'spam' else 0
    
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)
    
    #Use your test sample in this iteration to make the class prediction.
    class_predicted = clf.predict([testSample])[0]
    
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label:
        error_count += 1

#Print the error rate
error_rate = error_count / len(db)
print(f'LOO-CV error rate: {error_rate:.2f}')
