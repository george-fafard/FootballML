""" 
    This will be where the actual code for the svm classifier goes
"""
from FootballML.Dataset import cleaned_data as cd
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# read in the data and split
data_read_2009 = cd.read_game_data_from_files(2009)
data_read_2010 = cd.read_game_data_from_files(2010)
data2010clean = cd.clean_data(np.array(data_read_2010[0]))
data2009clean = cd.clean_data(np.array(data_read_2009[0]))
X, Y = cd.get_training(data2009clean, data2010clean, np.array(data_read_2010[0]), 2010)
X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

# params to "search"
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'sigmoid', 'linear']}

# train the model on train set
model = SVC()
model.fit(X_train, y_train)

# print prediction results
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

# n_jobs = -1 will maximize the use of your CPU, remove for slower but less taxing computations
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(x_test)

# print classification report
print(classification_report(y_test, grid_predictions))
