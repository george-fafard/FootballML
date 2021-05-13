"""
    Ensemble classifier model.

    This is to be imported in the testing notebook for the  
    ensemble classifier.
"""
# Data structures and manipulation
#import numpy  as np
#import pandas as pd

# Model and learning operations
#import sklearn.preprocessing as scalers
#import sklearn.metrics       as metrics
from sklearn.ensemble        import StackingClassifier
from sklearn.model_selection import train_test_split

# FootballML imports
from FootballML.Classifiers.Individual.logistic_regression_classifier import get_training_labels
from FootballML.Classifiers.Individual.logistic_regression_classifier import hyperparam_tuned_log_regression
from FootballML.Classifiers.Individual.neural_network_classifier      import hyperparam_tuned_neural_network
from FootballML.Classifiers.Individual.random_forest_classifier       import hyperparam_tuned_random_forest
from FootballML.Classifiers.Individual.svm.svm_classifier             import hyperparam_tuned_support_vector


def hyperparam_tuned_ensemble_classifier():
    """Ensemble classifier with custom hyperparameters.

    Returns
    -------
    sklearn StackingClassifier object
        The ensemble classifier with custom hyperparameters
    """
    # List of the individual classifiers to be used in the ensemble
    # classifier with their names
    estimators = [('Log Reg', hyperparam_tuned_log_regression()),
                  ('Nrl Net', hyperparam_tuned_neural_network()),
                  ('RForest', hyperparam_tuned_random_forest() ),
                  ('SVM'    , hyperparam_tuned_support_vector())] 

    # Ensemble classifier
    return StackingClassifier(estimators=estimators)


def run_ensemble_classifier():
    """Run the ensemble classifier on testing data to get the results.

    To be imported in the testing/results notebook.

    Returns
    -------
    none
    """
    # Training labels 
    X, Y = get_training_labels(start_year=2003, end_year=2019)

    # Training and testing data. Test size is the number of games in the test
    # sample. Setting the split to not be shuffled will cause the test sample
    # to be taken from the end of data. Thus, in this case the integer value 
    # for test size will be the number of games at the end of the data (with 15
    # games being used for each season). Here, I have it set to the last two seasons.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30, shuffle=False)

    # Fit classifier
    ensemble_classifier = hyperparam_tuned_ensemble_classifier()
    ensemble_classifier.fit(X, Y)

    # Test data predictions and accuracy score
    score = ensemble_classifier.score(X_test, Y_test)

    # Display metrics
    print('Score:', score)
