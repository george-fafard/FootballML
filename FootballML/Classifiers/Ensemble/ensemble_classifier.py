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
from FootballML.Classifiers.Individual.logistic_regression_classifier import hyperparam_tuned_log_regression
from FootballML.Classifiers.Individual.neural_network_classifier      import hyperparam_tuned_neural_network
from FootballML.Classifiers.Individual.random_forest_classifier       import hyperparam_tuned_random_forest
from FootballML.Classifiers.Individual.svm.svm_classifier             import hyperparam_tuned_support_vector


# Example 
# ------------------------
# 1) Load data in this file and split it here and do any necessary
#    scaling
#X_train, X_test, Y_train, Y_test = train_test_split()

# List of the individual classifiers to be used in the ensemble
# classifier with their names
estimators = [('Log Reg', hyperparam_tuned_log_regression()),
              ('Nrl Net', hyperparam_tuned_neural_network()),
              ('RForest', hyperparam_tuned_random_forest() ),
              ('SVM'    , hyperparam_tuned_support_vector())] 

# Ensemble classifier
ensemble_classifier = StackingClassifier(estimators=estimators)

# 5) Fit the data
# Call ensemble.fit(X, Y)

# 6) Test on the testing data
# Test ensemble on test data and get results
