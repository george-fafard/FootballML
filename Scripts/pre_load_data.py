"""
    Quick script to pre-load the raw data and save them to files
"""
# Insert top level directory into the system path so 
# parent and sibling modules can be imported
import sys
sys.path.insert(0, '../')

# Imports
from FootballML.Dataset.cleaned_data import save_game_data_to_files
