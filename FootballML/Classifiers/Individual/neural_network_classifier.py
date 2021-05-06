""" 
    This will be where the actual code for the neural network classifier goes
"""
# THIS IS AN EXAMPLE YOU CAN DELETE THIS WHEN YOU START WORKING ON IT.
from FootballML.Dataset.cleaned_data import read_game_data_from_files

# Get test data to import in the notebook
def test_data():
    return read_game_data_from_files(2008)
