""" 
    This will be where the actual code for the random forest classifier goes
"""
# THIS IS AN EXAMPLE YOU CAN DELETE THIS WHEN YOU START WORKING ON IT.
from FootballML.Dataset.cleaned_data import read_game_data_from_files
from FootballML.Dataset.cleaned_data import get_training
from FootballML.Dataset.cleaned_data import getTrain
from FootballML.Dataset import cleaned_data as cd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
from pprint import pprint
from sklearn import preprocessing as p
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def hyperparam_tuned_random_forest():
    """Random forest classifier with custom hyperparameters.

    This is to be imported and implemented in the ensemble
    classifier.

    Returns
    -------
    sklearn RandomForestClassifier object
        The random forest classifier with custom hyperparameters
    """
    return RandomForestClassifier(bootstrap=False   , max_depth=900       , max_features='sqrt', 
                                  min_samples_leaf=7, min_samples_split=11, n_estimators=800)


# Get test data to import in the notebook
def test_data():
    return read_game_data_from_files(2009)

def test_train():
    return getTrain()
'''
Load the data in from 2003 to 2019
        Parameters
        ----------
        None
        
        Returns
        ------- 
        xtrain,xtest,ytrain,ytest
           Processes all the data and returns values to be implemented
'''
def get_data():
    data_read = cd.read_game_data_from_files(2003, 2019)
    # create big X and big Y
    for i in range(0, (16)):
        if i == 0:
            # Call functions in clean data to read and prep all data
            X, Y = cd.get_training(cd.clean_data(np.array(data_read[i])), cd.clean_data(np.array(data_read[i+1])),
                                   np.array(data_read[i+1]), 2003+i)
        else:
            X_temp, Y_temp = cd.get_training(cd.clean_data(np.array(data_read[i])), cd.clean_data(np.array(data_read[i+1])),
                                   np.array(data_read[i+1]), 2003+i)
            X += X_temp
            Y += Y_temp
    
    normalized_X = preprocessing.normalize(X)

    # standardize the data
    standardized_X = preprocessing.scale(X)
    #split and return data into testing and training
    xtrain,xtest,ytrain,ytest= train_test_split(standardized_X,Y,shuffle=False, test_size=0.3)
    return xtrain,xtest,ytrain,ytest
# Test to make sure it works for simple random forest
def test_run():
    xtrain,xtest,ytrain,ytest = get_data()
    clf3 = RandomForestClassifier()
    clf3.fit(xtrain, ytrain)
    clf3.predict(xtest)
    return clf3.score(xtest,ytest)
'''
Gets the best parameters possible for the random forest algorithm. This is done by feeding the random searchcv a lot of options 
as seen below. It will do many runs to find the best combination, takes a while to run.
        Parameters
        ----------
        None
        
        Returns
        ------- 
        best_params_ which tells you what the best model's parameters look like.
'''
def get_params():
    xtrain,xtest,ytrain,ytest = get_data()
    model = RandomForestRegressor(random_state = 42)
    
    #num of tress
    n_estimators = [100,300,600,800,1200,1600,2000]

    # num of features to look at
    max_features = ['auto', 'sqrt','log2']
    
    #whether bootstrap is used
    bootstrap = [True, False]

    # max depth of tree
    max_depth = [100,300,500,800,1000,1200,1400]
    
    #min num of samples to be a leaf node
    min_samples_leaf = [1, 2, 4,6,7]


    # required num to split internal node
    min_samples_split = [1,2, 5, 10,12,15]

    #combine all params into grid
    grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    model = RandomForestRegressor()

    model_random = RandomizedSearchCV(estimator = model, param_distributions = grid, n_iter = [50,100,150], cv = 3, verbose=2, random_state=42, n_jobs = -1)

    model_random.fit(xtrain, ytrain)
    return model_random.best_params_

'''
Uses the best parameters found from get_params() and runs a GridSearchCV on them. Other numbers closer to the best params
are used to make sure you are using the best params to get the highest accuracy. 
        Parameters
        ----------
        None
        
        Returns
        ------- 
        accuracy of the model
           Processes all the data and returns values to be implemented
'''
def gridCV():
    xtrain,xtest,ytrain,ytest = get_data()
    #best paramaeters 
    param_grid2 = {
    'bootstrap': [True],
    'max_depth': [700,800,900],
    'max_features': ['sqrt'],
    'min_samples_leaf': [6,7,8],
    'min_samples_split': [9,10,11],
    'n_estimators': [700,800,900]
}
    grid_rf =  RandomForestRegressor()
    gridCV = GridSearchCV(estimator = grid_rf, param_grid = param_grid2, 
                          cv = 3, n_jobs = -1, verbose = 2)
    gridCV.fit(xtrain,ytrain)
    return performance_accuracy(ytest,xtest, gridCV)
'''
Uses the best parameters found from get_params() and runs a GridSearchCV on them. Other numbers closer to the best params
are used to make sure you are using the best params to get the highest accuracy. 
        Parameters
        ----------
        ytest - ytest data
        xtest - xtest data
        model - the model you are using 
        
        Returns
        ------- 
        nothing, prints accracy
           
'''
def performance_accuracy(ytest,xtest, model):

    test = abs(abs(np.round(model.predict(xtest),0)) - ytest)
    accuracy = (test==0).sum() / len(test) * 100


    print('Accuracy:', round(accuracy, 2), '%.') 
