"""
    Ensemble classifier model.

    This is to be imported in the testing notebook for the  
    ensemble classifier.
"""
# Data structures and manipulation
import numpy  as np
import pandas as pd

# Model and learning operations
import sklearn.preprocessing as scalers
import sklearn.metrics       as metrics
from sklearn.ensemble import StackingClassifier

# FootballML imports
from FootballML.Dataset.cleaned_data import read_game_data_from_files 
from FootballML.Classifiers.Individual.logistic_regression_classifier import logistic_regression_classifier


# Test 
log_reg_classifier = logistic_regression_classifier()
def test():
    print('Test importing ensemble classifier stuff')

"""
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

test1 = LogisticRegression()
test2 = SVC()

X_train, X_test, Y_train, Y_test = get_labels_test()

estimators = [('test1', test1), ('test2', test2)]

stack = StackingClassifier(estimators=estimators)
stack.fit(X_train, Y_train)

stack.score(X_test, Y_test)
"""
