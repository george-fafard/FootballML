import os
import pathlib

# Root directory
ROOT_DIRECTORY = str(pathlib.Path(os.path.dirname(os.path.abspath(__file__))))

# Set data path
DATA_PATH = str(pathlib.Path(ROOT_DIRECTORY + '/Dataset/Loaded_Raw_Data/'))
