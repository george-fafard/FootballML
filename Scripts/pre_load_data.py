"""
    Quick script to pre-load the raw data and save them to files
"""
# Insert top level directory into the system path so parent and sibling 
# modules can be imported 
import pathlib
import sys

# Convert path to use proper directory markers based on 
# operating system
PATH = str(pathlib.Path('../'))

# Insert path to system path
sys.path.insert(0, PATH)

# Imports
from FootballML import DATA_PATH
from FootballML.Dataset.cleaned_data import save_game_data_to_files


# Settings
START_YEAR = 2003
END_YEAR   = 2019

# Save the data
save_game_data_to_files(START_YEAR, END_YEAR)
