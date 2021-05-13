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
from FootballML.Classifiers.Individual.logistic_regression_classifier import hyperparam_tuned_log_reg_classifier


# Example 
# ------------------------
# 1) Load data in this file and split it here and do any necessary
#    scaling
#X_train, X_test, Y_train, Y_test = train_test_split()

# 2) Load our classifiers
log_reg_classifier = hyperparam_tuned_log_reg_classifier()
# other classifiers...

# 3) Add the classifiers to be used in the ensemble model
estimators = [('Log Reg', log_reg_classifier)] # Once other classifiers added: --> 
                                               #      [('Log Reg', log_reg_classifier), ('others'..., others...)]

# 4) Create the ensemble model using our classifiers list we created
ensemble = StackingClassifier(estimators=estimators)

# 5) Fit the data
# Call ensemble.fit(X, Y)

# 6) Test on the testing data
# Test ensemble on test data and get results
