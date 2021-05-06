"""
    This will be where the actual code for the ensemble classifier goes.
"""
# THIS IS AN EXAMPLE. WE CAN REMOVE IT WHEN WE START WORKING ON THIS
from FootballML.Dataset.cleaned_data import read_game_data_from_files 

# Get test data
def test_data():
    return read_game_data_from_files(2005, 2010) 
