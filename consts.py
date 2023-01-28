# library imports
import os

RESULTS_FOLDER_NAME = "results"
DATA_FOLDER_NAME = "data"
RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME)
DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)

SETUP_FOLDERS = [RESULTS_FOLDER_PATH, DATA_FOLDER_PATH]