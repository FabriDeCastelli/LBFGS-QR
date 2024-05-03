import os

LBFGS_RESULTS = "/testing/results/LBFGS/{}/{}"
QR_RESULTS = "/testing/results/QR/{}"
CONDITIONING_RESULTS = "/testing/results/conditioning/{}"
COMPARISON_RESULTS = "/testing/results/comparison/{}"
PLOTS_PATH = "/testing/plots/{}/{}"

# PROJECT FOLDER PATH
PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LBFGS_RESULTS = PROJECT_FOLDER_PATH + LBFGS_RESULTS
QR_RESULTS = PROJECT_FOLDER_PATH + QR_RESULTS
CONDITIONING_RESULTS = PROJECT_FOLDER_PATH + CONDITIONING_RESULTS
COMPARISON_RESULTS = PROJECT_FOLDER_PATH + COMPARISON_RESULTS
PLOTS_PATH = PROJECT_FOLDER_PATH + PLOTS_PATH

# PLOTTING CONFIGURATIONS
from matplotlib import pyplot as plt
import seaborn as sns


def setup(title=None, x_label=None, y_label=None):
    sns.set_style("whitegrid")
    sns.set_context('paper')
    plt.figure(figsize=(13, 9))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=20, pad=20, loc='center', fontweight='bold')
    plt.xlabel(x_label, fontsize=15, labelpad=20)
    plt.ylabel(y_label, fontsize=15, labelpad=20)


def nanoseconds_to_seconds(x, pos=None):
    return f"{x / 10 ** 9:.4f}"


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


def nanoseconds_to_milliseconds(x, pos=None):
    return f"{x / 10 ** 6:.1f}"

def scientific_notation(x, pos):
    return '%1.1e' % x
