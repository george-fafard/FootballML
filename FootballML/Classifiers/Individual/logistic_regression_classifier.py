""" 
    Individual Logistic Regression model.

    This is to be imported in the testing notebook for Logistic 
    Regression and used as part of the ensemble classifier.
"""
# Data structures and manipulation
import numpy  as np

# Model and learning operations
import sklearn.preprocessing as scalers
import sklearn.metrics       as metrics
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split

# FootballML imports
from FootballML.Dataset import cleaned_data as cd


def get_training_labels(start_year, end_year):
    """Get the training labels from all years.

    Parameters
    ----------
    start_year : int
        Previous year to the first year to use
    end_year : int
        The last year to use

    Returns
    -------
    list, list
        X, Y --> Training labels
    """
    # List of game data with each index containing the game data
    # for a year. Each year is converted to a numpy array for 
    # getting the training data below
    game_data = cd.read_game_data_from_files(start_year, end_year)
    game_data = [np.array(year) for year in game_data]

    # Training labels
    X, Y = [], []

    # Prepare the training labels from each year. Start year is not
    # included in the training data and is only used as the previous
    # year to the first year. The first year is the next year following
    # the specified start year
    for prev_year in range(len(game_data) - 1):
        # Arguments
        PREVOUS_YEAR_CLEAN = cd.clean_data(game_data[prev_year])
        CURRENT_YEAR_CLEAN = cd.clean_data(game_data[prev_year + 1])
        CURRENT_YEAR_RAW   = game_data[prev_year + 1]
        CURRENT_YEAR_DIGIT = start_year + (prev_year + 1)

        # Extract training labels for current year
        X_YEAR, Y_YEAR = cd.get_training(PREVOUS_YEAR_CLEAN, CURRENT_YEAR_CLEAN, CURRENT_YEAR_RAW, CURRENT_YEAR_DIGIT)
        
        # Add training labels to those from the previous years
        X.extend(X_YEAR)
        Y.extend(Y_YEAR)

    return X, Y


def scale_features(X, Y, name='Standard'):
    """Scale features for training and testing.

    Parameters
    ----------
    X : list
        Feature labels
    Y : list
        Target labels
    name : str
        Name of the scale to use

    Returns
    -------
    list
        Scaled feature labels
    """
    # Select feature scaler
    if name == 'MinMax':
        scaler = scalers.MinMaxScaler()
    elif name == 'Robust':
        scaler = scalers.RobustScaler()
    elif name == 'Quantile':
        scaler = scalers.QuantileTransformer()
    elif name == 'Power':
        scaler = scalers.PowerTransformer()
    elif name == 'Standard':
        scaler = scalers.StandardScaler()
    else:
        print('Invalid scale name, defaulting to Standard')
        scaler = scalers.StandardScaler()

    # Scale features
    return scaler.fit_transform(X, Y)

def hyperparam_tuned_log_regression():
    """Logistic regression classifier with custom hyperparameters.

    This is to be imported and implemented in the ensemble
    classifier.

    Returns
    -------
    sklearn LogisticRegression object
        The logistic regression classifier with custom hyperparameters
    """
    return LogisticRegression(max_iter=1000000)


def display_metrics(classifier, X_test, Y_test):
    """Display results from running the classifier on testing data

    Parameters
    ----------
    classifier : sklearn classifier object
        Classifier to test

    Returns
    -------
    none
    """
    # Test data predictions and accuracy score
    Y_pred = classifier.predict(X_test)
    score  = classifier.score(X_test, Y_test)

    # Test data confusion matrix
    conf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

    # Display metrics
    print('Score:', score)
    print()
    print('Confusion matrix:')
    print(conf_matrix)

    # Precision-Recall curve
    curve = metrics.plot_precision_recall_curve(classifier, X_test, Y_test)
    curve.ax_.set_title('Game winner prediction Precision-Recall curve')


def run_logistic_regression():
    """Run the logistic regression classifier on testing data to get the results.

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
    log_reg_classifier = hyperparam_tuned_log_regression()
    log_reg_classifier.fit(X_train, Y_train)

    # Run the classifier on testing data and display the results
    display_metrics(log_reg_classifier, X_test, Y_test)
