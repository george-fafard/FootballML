"""
    Ensemble classifier model.

    This is to be imported in the testing notebook for the  
    ensemble classifier.
"""
# Model and learning operations
from sklearn.ensemble        import StackingClassifier
from sklearn.model_selection import train_test_split

# FootballML imports
from FootballML.Classifiers.Individual.logistic_regression_classifier import get_training_labels
from FootballML.Classifiers.Individual.logistic_regression_classifier import scale_features
from FootballML.Classifiers.Individual.logistic_regression_classifier import hyperparam_tuned_log_regression
from FootballML.Classifiers.Individual.logistic_regression_classifier import display_metrics
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

    # Scaled feature labels
    X_scaled = scale_features(X, Y, name='Quantile')

    # Training and testing split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.15, shuffle=False)

    # Fit classifier
    ensemble_classifier = hyperparam_tuned_ensemble_classifier()
    ensemble_classifier.fit(X_train, Y_train)

    # Run the classifier on testing data and display the results
    display_metrics(ensemble_classifier, X_test, Y_test)
