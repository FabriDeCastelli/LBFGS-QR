import os

LBFGS_RESULTS = "/testing/resultsLBFGS/{}"
QR_RESULTS = "/testing/resultsQR/{}"
PLOTS_PATH = "/testing/plots/{}"

# PROJECT FOLDER PATH
PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LBFGS_RESULTS = PROJECT_FOLDER_PATH + LBFGS_RESULTS
QR_RESULTS = PROJECT_FOLDER_PATH + QR_RESULTS
PLOTS_PATH = PROJECT_FOLDER_PATH + PLOTS_PATH