""" 
    Individual logistic regression model.

    This is to be imported in the testing notebook for logistic 
    regression and used as part of the ensemble classifier.
"""
# Library imports
import numpy  as np
import pandas as pd
from sklearn                 import preprocessing as p
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import train_test_split

# FootballML imports
from FootballML.Dataset import cleaned_data as cd


def run_logistic_regression():
    """To be imported in the testing notebook.

    Returns
    -------
    none
    """
    # Year range
    START_YEAR = 2003
    END_YEAR   = 2019

    # List of game data with each index containing the game data
    # for a year. Each year is converted to a numpy array for 
    # getting the training data below
    game_data = cd.read_game_data_from_files(START_YEAR, END_YEAR)
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
        CURRENT_YEAR_DIGIT = START_YEAR + (prev_year + 1)

        # Extract training labels for current year
        X_YEAR, Y_YEAR = cd.get_training(PREVOUS_YEAR_CLEAN, CURRENT_YEAR_CLEAN, CURRENT_YEAR_RAW, CURRENT_YEAR_DIGIT)
        
        # Add training labels to those for the previous years
        X.extend(X_YEAR)
        Y.extend(Y_YEAR)

    # Fearure scale (uncomment scale to use)
    scaler = p.MinMaxScaler()
    #scaler = p.RobustScaler()
    #scaler = p.QuantileTransformer()
    #scaler = p.PowerTransformer()
    #scaler = p.StandardScaler()

    # Scale the features
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Training and testing data. Test size is the number of games in the test
    # sample. Setting the split to not be shuffled will cause the test sample
    # to be taken from the end of data. Thus, in this case the integer value 
    # for test size will be the number of games at the end of the data (with 15
    # games being used for each season). Here, I have it set to the last two seasons.
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=30, shuffle=False)

    # Fit classifier
    log_reg_classifier = LogisticRegression(max_iter=1000000)
    log_reg_classifier.fit(X_train, Y_train)

    # Get score
    score = log_reg_classifier.score(X_test, Y_test)
    print("Score:", score)

    # Confusion matrix
    Y_pred = log_reg_classifier.predict(X_test)
    log_conf_matrix = confusion_matrix(Y_test, Y_pred)
    print('\nConfusion matrix:')
    print(log_conf_matrix)
