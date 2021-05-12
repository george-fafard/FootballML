""" 
    This will be where the actual code for the logistic regression classifier goes
"""
# Library imports
import numpy as np
import pandas as pd

# FootballML imports
from FootballML.Dataset import cleaned_data as cd


def run_logistic_regression():
    """To be imported in the testing notebook.

    Returns
    -------
    none
    """
    # List of game data with each index containing the game data
    # for a year
    game_data = cd.read_game_data_from_files(start_year=2018, end_year=2019)

    for year in game_data:
        year = np.array(year)


