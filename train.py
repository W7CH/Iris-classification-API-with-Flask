#Import libraries
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn import datasets

import pickle #take any python object and save it as binary, so that it can be loaded

#Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=75, train_size=0.75)

#Build and train using SVC
model_svc = SVC(gamma='auto')
model_svc.fit(X_train, y_train)

#Find model's accuracy
accuracy_svc = model_svc.score(X_test, y_test)
print("Accuracy for SVC is: {}".format(accuracy_svc))

#Build and train using RFC
model_rfc = RandomForestClassifier(n_estimators=25)
model_rfc.fit(X_train, y_train)

#Find model's accuracy
accuracy_rfc = model_rfc.score(X_test, y_test)
print("Accuracy for RFC is: {}".format(accuracy_rfc))

#Generate binary files
with open("model_svc.pkl", "wb") as model_svc_pickle:
    pickle.dump(model_svc, model_svc_pickle)
with open("model_rfc.pkl", "wb") as model_rfc_pickle:
    pickle.dump(model_rfc, model_rfc_pickle)
