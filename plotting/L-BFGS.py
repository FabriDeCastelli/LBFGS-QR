from config import LBFGS_RESULTS, PLOTS_PATH

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, cm
import seaborn as sns

metrics_map = {
    'error': ['relative', 'residual'],
    'time': ['meantime'],
    'iterations': ['iterations']
}


def nanoseconds_to_milliseconds(x, pos):
    return f"{x / 10 ** 6:.1f}"


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


def plot_lambda_epsilon_3D(kind):
    if kind not in ['error', 'time', 'iterations']:
        raise ValueError("kind must be one of 'error', 'time', 'iterations'")

    df = pd.read_csv(LBFGS_RESULTS.format("statisticsLBFGS-lambda-eps-m300n20--{}.csv".format(kind)))

    metrics = metrics_map[kind]

    for metric in metrics:

        lambda_values = df['lambda']
        epsilon_values = df['epsilon']
        meantime_values = df[metric]

        lambda_mesh, epsilon_mesh = np.meshgrid(np.unique(lambda_values), np.unique(epsilon_values))

        # Cubic interpolation of the data to make it smooth
        meantime_interp = griddata((lambda_values, epsilon_values), meantime_values, (lambda_mesh, epsilon_mesh),
                                   method='cubic')

        # Plot the 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sns.set(style="white", context="paper")

        surface = ax.plot_surface(np.log10(lambda_mesh), np.log10(epsilon_mesh), meantime_interp, edgecolor='none',
                                  cmap='viridis', alpha=0.9)

        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.2)

        # Set labels
        ax.set_xlabel('λ', labelpad=10, fontsize=10)
        ax.set_ylabel('ϵ', labelpad=10, fontsize=10)

        if metric == 'meantime':
            metric = 'running time'
            ax.set_zlabel('time (ms)', labelpad=10, fontsize=10)
            ax.zaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_milliseconds))
        elif metric in ['relative', 'residual']:
            metric = metric + ' error'
            ax.set_zlabel(metric, labelpad=10, fontsize=10)
        else:
            ax.set_zlabel(metric, labelpad=10, fontsize=10)

        # Set logarithmic locator of ticks
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

        plt.title('L-BFGS ' + metric + ' with respect to λ and ϵ over 10000 runs', fontsize=15)

        metric = metric.replace(" ", "_")
        plt.savefig(PLOTS_PATH.format("LBFGS-lambda-eps--{}.png".format(metric)))
        plt.show()


def plot_epsilon_memory(kind):
    if kind not in ['error', 'iterations']:
        raise ValueError("kind must be one of 'error', 'iterations'")

    df = pd.read_csv(LBFGS_RESULTS.format("statisticsLBFGS-eps-mem-m300n20--{}.csv".format(kind)))

    metrics = metrics_map[kind]

    for metric in metrics:

        memory_values = df['memsize']
        epsilon_values = df['epsilon']
        meantime_values = df[metric]

        memory_mesh, epsilon_mesh = np.meshgrid(np.unique(memory_values), np.unique(epsilon_values))

        # Cubic interpolation of the data to make it smooth
        meantime_interp = griddata((memory_values, epsilon_values), meantime_values, (memory_mesh, epsilon_mesh),
                                   method='cubic')

        # Plot the 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sns.set(style="white", context="paper")

        surface = ax.plot_surface(memory_mesh, np.log10(epsilon_mesh), meantime_interp, edgecolor='none',
                                  cmap='viridis', alpha=0.9)

        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.2)

        # Set labels
        ax.set_xlabel('memory size', labelpad=10, fontsize=10)
        ax.set_ylabel('ϵ', labelpad=10, fontsize=10)

        ax.set_zlabel('iterations', labelpad=10, fontsize=10)

        if metric in ['relative', 'residual']:
            metric = metric + ' error'
            ax.set_zlabel(metric, labelpad=10, fontsize=10)

        # Set logarithmic locator of ticks
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

        plt.title('L-BFGS ' + metric + ' with respect to the memory size and ϵ over 10000 runs', fontsize=15)

        plt.show()
        plt.savefig(PLOTS_PATH.format("LBFGS-eps-mem--{}.png".format(metric)))


if __name__ == "__main__":
    plot_lambda_epsilon_3D('error')
    plot_lambda_epsilon_3D('time')
    plot_lambda_epsilon_3D('iterations')
    plot_epsilon_memory('error')
    plot_epsilon_memory('iterations')
