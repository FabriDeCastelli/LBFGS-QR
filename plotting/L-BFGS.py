from config import (
    LBFGS_RESULTS,
    PLOTS_PATH,
    log_tick_formatter,
    nanoseconds_to_seconds,
    setup
)

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, cm
import seaborn as sns
from scipy.interpolate import interp1d

metrics_map = {
    'error': ['relative', 'residual'],
    'time': ['meantime'],
    'iterations': ['iterations']
}


def plot_lambda_epsilon_3D(kind):
    if kind not in ['error', 'time', 'iterations']:
        raise ValueError("kind must be one of 'error', 'time', 'iterations'")

    df = pd.read_csv(LBFGS_RESULTS.format("3D", "statisticsLBFGS-lambda-eps-m300n20--{}.csv".format(kind)))

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
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(111, projection='3d')

        sns.set(style="white", context="paper")

        surface = ax.plot_surface(np.log10(lambda_mesh), np.log10(epsilon_mesh), meantime_interp, edgecolor='none',
                                  cmap='viridis', alpha=0.9)

        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.2)

        # Set labels
        ax.set_xlabel('λ', labelpad=20, fontsize=15)
        ax.set_ylabel('ϵ', labelpad=20, fontsize=15)

        if metric == 'meantime':
            metric = 'running time'
            ax.set_zlabel('time (s)', labelpad=10, fontsize=20)
            ax.zaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_seconds))
            title = 'L-BFGS ' + metric + ' with respect to λ and ϵ over 10000 runs'
        elif metric in ['relative', 'residual']:
            if metric == 'relative':
                metric = 'relative error'
            ax.set_zlabel(metric, labelpad=10, fontsize=20)
            title = 'L-BFGS ' + metric + ' with respect to λ and ϵ'
        else:
            ax.set_zlabel(metric, labelpad=10, fontsize=20)
            title = 'L-BFGS ' + metric + ' with respect to λ and ϵ'

        plt.title(title, fontsize=20, pad=20, fontweight='bold', loc='center')

        # Set logarithmic locator of ticks
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

        ax.xaxis.set_tick_params(size=15)
        ax.yaxis.set_tick_params(size=15)
        ax.zaxis.set_tick_params(size=15)

        metric = metric.replace(" ", "_")
        plt.savefig(PLOTS_PATH.format("LBFGS/3D", "LBFGS-lambda-eps--{}.png".format(metric)))
        plt.show()


def plot_epsilon_memory_3D(kind):
    if kind not in ['error', 'iterations']:
        raise ValueError("kind must be one of 'error', 'iterations'")

    df = pd.read_csv(LBFGS_RESULTS.format("3D", "statisticsLBFGS-eps-mem-m300n20--{}.csv".format(kind)))

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
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(111, projection='3d')

        sns.set(style="white")
        sns.set_context('paper')

        surface = ax.plot_surface(memory_mesh, np.log10(epsilon_mesh), meantime_interp, edgecolor='none',
                                  cmap='viridis', alpha=0.9)

        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.2)

        # Set labels
        ax.set_xlabel('memory size', labelpad=20, fontsize=15)
        ax.set_ylabel('ϵ', labelpad=20, fontsize=15)
        ax.set_zlabel('iterations', labelpad=10, fontsize=20)

        if metric in ['relative', 'residual']:
            if metric == 'relative':
                metric = 'relative error'
            ax.set_zlabel(metric, labelpad=10, fontsize=20)

        plt.title('L-BFGS ' + metric + ' with respect to the memory size and ϵ', fontsize=20, pad=20,
                  fontweight='bold',
                  loc='center')

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

        ax.xaxis.set_tick_params(size=15)
        ax.yaxis.set_tick_params(size=15)
        ax.zaxis.set_tick_params(size=15)
        plt.savefig(PLOTS_PATH.format("LBFGS/3D", "LBFGS-eps-mem--{}.png".format(metric)))
        plt.show()


def plot_iterations_error_2D(conditioning="well"):
    if conditioning not in ["well", "ill"]:
        raise ValueError("conditioning must be one of 'well' or 'ill'")

    setup(
        title=f'L-BFGS errors with respect to iterations on {conditioning}-conditioned matrix',
        x_label='Iterations',
        y_label='Error'
    )
    conditioning = conditioning + "_conditioned"

    df = pd.read_csv(LBFGS_RESULTS.format(conditioning, "statisticsLBFGS-iterations-m1000n20--error.csv"))

    maxiterations_values = df['maxiterations']
    relative_values = df['relative']
    residual_values = df['residual']

    # since from a certain row the residual and the relative gap are the same as the previous row, we can all the
    # next rows and keep only the first row of each value
    relative_values = relative_values.drop_duplicates(keep='first')
    residual_values = residual_values.drop_duplicates(keep='first')

    assert len(relative_values) == len(residual_values)

    maxiterations_values = maxiterations_values[:len(relative_values)]

    max_iter_interp = np.linspace(min(maxiterations_values), max(maxiterations_values), 100)

    relative_interp = interp1d(maxiterations_values, relative_values, kind='cubic')(max_iter_interp)
    residual_interp = interp1d(maxiterations_values, residual_values, kind='cubic')(max_iter_interp)

    data = pd.DataFrame({
        'maxiterations': max_iter_interp,
        'relative': relative_interp,
        'residual': residual_interp
    })

    sns.lineplot(data=data, x='maxiterations', y='relative', label='Relative Error', color='blue')
    sns.lineplot(data=data, x='maxiterations', y='residual', label='Residual', color='red')

    plt.legend(loc='upper right', fontsize=15)
    plt.ylim(0, None)

    conditioning = "LBFGS/" + conditioning

    plt.savefig(PLOTS_PATH.format(conditioning, "LBFGS-iterations-error.png"))
    plt.show()


def plot_norm_gradient_2D(conditioning="well", smoothing_window=5):
    if conditioning not in ["well", "ill"]:
        raise ValueError("conditioning must be one of 'well' or 'ill'")
    cond = conditioning
    setup(
        title=f'L-BFGS convergence on {cond}-conditioned matrix',
        x_label='Iterations',
        y_label=' '
    )

    conditioning = conditioning + "_conditioned"

    df = pd.read_csv(LBFGS_RESULTS.format(conditioning, "statisticsLBFGS-iterations-m1000n20--error-norm.csv"))

    gradnorm = df['mean_gradient'].unique()
    error = df['mean_relative'].unique()
    residual = df['mean_residual'].unique()

    iterations = list(range(len(gradnorm)))

    assert len(error) == len(residual) == len(gradnorm)

    iterations_interp = np.linspace(min(iterations), max(iterations), len(residual))
    gradnorm = interp1d(iterations, gradnorm, kind='quadratic')(iterations_interp)
    error = interp1d(iterations, error, kind='quadratic')(iterations_interp)
    residual = interp1d(iterations, residual, kind='quadratic')(iterations_interp)

    smoothed_gradnorm = np.convolve(gradnorm, np.ones(smoothing_window) / smoothing_window, mode='valid')
    smoothed_error = np.convolve(error, np.ones(smoothing_window) / smoothing_window, mode='valid')
    smoothed_residual = np.convolve(residual, np.ones(smoothing_window) / smoothing_window, mode='valid')

    data = pd.DataFrame({
        'iterations': iterations[:len(smoothed_gradnorm)],
        'Gradient Norm': smoothed_gradnorm,
        'Relative Error': smoothed_error,
        'Residual': smoothed_residual
    })

    sns.lineplot(data=data, x='iterations', y='Gradient Norm', label='Gradient Norm', color='green')
    sns.lineplot(data=data, x='iterations', y='Relative Error', label='Relative Error', color='blue')
    sns.lineplot(data=data, x='iterations', y='Residual', label='Residual', color='red')

    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=15)

    conditioning = "LBFGS/" + conditioning
    plt.savefig(PLOTS_PATH.format(conditioning, f"LBFGS-iterations-gradient-{cond}.png"))
    plt.show()


def plot_iteration_memory(conditioning="well", smoothing_window=3):
    if conditioning not in ["well", "ill"]:
        raise ValueError("conditioning must be one of 'well' or 'ill'")

    cond = conditioning
    setup(title=f'L-BFGS relative error for different memory sizes on {cond}-conditioned matrix', x_label='Iterations', y_label='Relative Error')

    conditioning = conditioning + "_conditioned"

    df = pd.read_csv(
        LBFGS_RESULTS.format(conditioning, "statisticsLBFGS-iterations-m1000n20--memsize.csv"),
    )

    unique_memsize = df['memsize'].unique()

    for memsize in unique_memsize:
        data = df[df['memsize'] == memsize]
        data.reset_index(drop=True, inplace=True)

        gradients_indices = data.index[data['gradient'].drop_duplicates(keep='first').index]
        relative_errors = data.loc[gradients_indices]['relative'].values

        extension = np.full(smoothing_window, relative_errors[-1])
        relative_errors = np.concatenate((relative_errors, extension))

        print(len(relative_errors))

        iterations = list(range(len(relative_errors)))
        iterations_interp = np.linspace(min(iterations), max(iterations), len(relative_errors))
        relative_interp = interp1d(iterations, relative_errors, kind='cubic')(iterations_interp)

        smoothed_relative = np.convolve(relative_interp, np.ones(smoothing_window) / smoothing_window, mode='valid')

        data = pd.DataFrame({
            'iterations': iterations_interp[:len(smoothed_relative)],
            'relative': smoothed_relative
        })

        sns.lineplot(data=data, x='iterations', y='relative', label='Memory Size = {}'.format(memsize))

    plt.legend(loc='upper right', fontsize=15)
    plt.yscale('log')

    conditioning = "LBFGS/" + conditioning
    plt.savefig(PLOTS_PATH.format(conditioning, f"LBFGS-iterations-memory-{cond}.png"))
    plt.show()


def plot_awls_vs_exact_gradient_decrease_2D(smoothing_window=3):
    conditioning = "well_conditioned"
    df2_awls_quad = pd.read_csv(
        LBFGS_RESULTS.format(conditioning, "statisticsLBFGS-AWLS-quad-iterations-m1000n20--error-norm.csv"),
    )
    df3_exact = pd.read_csv(
        LBFGS_RESULTS.format(conditioning, "statisticsLBFGS-iterations-m1000n20--error-norm.csv"),
    )

    setup(title='L-BFGS ExactLS vs AWLS gradient norm on well-conditioned matrix', x_label='Iterations', y_label='Gradient Norm')

    gradnorm_awls_quad = df2_awls_quad['mean_gradient'].unique()
    gradnorm_exact = df3_exact['mean_gradient'].unique()

    extension = np.full(len(gradnorm_awls_quad) - len(gradnorm_exact) + smoothing_window, gradnorm_exact[-1])
    gradnorm_exact = np.concatenate((gradnorm_exact, extension))

    extension = np.full(smoothing_window, gradnorm_awls_quad[-1])
    gradnorm_awls_quad = np.concatenate((gradnorm_awls_quad, extension))

    assert len(gradnorm_awls_quad) == len(gradnorm_exact)

    iterations = list(range(len(gradnorm_awls_quad)))

    iterations_interp = np.linspace(min(iterations), max(iterations), len(gradnorm_awls_quad))

    gradnorm_awls_quad = interp1d(iterations, gradnorm_awls_quad, kind='cubic')(iterations_interp)
    gradnorm_exact = interp1d(iterations, gradnorm_exact, kind='cubic')(iterations_interp)

    smoothed_gradnorm_awls_quad = np.convolve(gradnorm_awls_quad, np.ones(smoothing_window) / smoothing_window,
                                              mode='valid')
    smoothed_gradnorm_exact = np.convolve(gradnorm_exact, np.ones(smoothing_window) / smoothing_window, mode='valid')

    data = pd.DataFrame({
        'iterations': iterations_interp[:len(smoothed_gradnorm_awls_quad)],
        'AWLS quadratic': smoothed_gradnorm_awls_quad,
        'ExactLS': smoothed_gradnorm_exact
    })

    sns.lineplot(data=data, x='iterations', y='AWLS quadratic', label='AWLS', color='blue')
    sns.lineplot(data=data, x='iterations', y='ExactLS', label='ExactLS', color='green')

    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=15)

    conditioning = "LBFGS/" + conditioning
    plt.savefig(PLOTS_PATH.format(conditioning, "LBFGS-LS-gradient-comparison.png"))
    plt.show()


def BFGS_comparison():
    folder = "comparison_BFGS"
    df_BFGS = pd.read_csv(
        LBFGS_RESULTS.format(folder, "statisticsBFGS-iterations-m1000n20--error-norm.csv"),
    )
    df_LBFGS = pd.read_csv(
        LBFGS_RESULTS.format("well_conditioned", "statisticsLBFGS-iterations-m1000n20--error-norm.csv"),
    )

    setup(title='BFGS vs L-BFGS gradient norm on well-conditioned matrix', x_label='Iterations', y_label=' ')

    gradnorm_BFGS = df_BFGS['mean_gradient'][::2]
    gradnorm_LBFGS = df_LBFGS['mean_gradient'][::2]
    error_BFGS = df_BFGS['mean_relative'][::2]
    error_LBFGS = df_LBFGS['mean_relative'][::2]
    residual_BFGS = df_BFGS['mean_residual'][::2]
    residual_LBFGS = df_LBFGS['mean_residual'][::2]

    min_len = 27

    iterations = list(range(min_len))
    gradnorm_BFGS = gradnorm_BFGS[:min_len]
    gradnorm_LBFGS = gradnorm_LBFGS[:min_len]
    error_BFGS = error_BFGS[:min_len]
    error_LBFGS = error_LBFGS[:min_len]
    residual_BFGS = residual_BFGS[:min_len]
    residual_LBFGS = residual_LBFGS[:min_len]

    iterations_interp = np.linspace(min(iterations), max(iterations), len(residual_BFGS))

    gradnorm_BFGS = interp1d(iterations, gradnorm_BFGS, kind='cubic')(iterations_interp)
    gradnorm_LBFGS = interp1d(iterations, gradnorm_LBFGS, kind='cubic')(iterations_interp)
    error_BFGS = interp1d(iterations, error_BFGS, kind='cubic')(iterations_interp)
    error_LBFGS = interp1d(iterations, error_LBFGS, kind='cubic')(iterations_interp)
    residual_BFGS = interp1d(iterations, residual_BFGS, kind='cubic')(iterations_interp)
    residual_LBFGS = interp1d(iterations, residual_LBFGS, kind='cubic')(iterations_interp)

    data = pd.DataFrame({
        'Iterations': iterations_interp,
        'BFGS Gradient': gradnorm_BFGS,
        'LBFGS Gradient': gradnorm_LBFGS,
        'BFGS Error': error_BFGS,
        'LBFGS Error': error_LBFGS,
        'BFGS Residual': residual_BFGS,
        'LBFGS Residual': residual_LBFGS
    })

    data_long = pd.melt(data, id_vars=['Iterations'], var_name='algorithm', value_name='value')

    # Define custom color palette
    custom_palette = {
        'BFGS Gradient': 'red',
        'BFGS Error': 'orange',
        'BFGS Residual': 'gold',
        'LBFGS Gradient': 'blue',
        'LBFGS Error': 'green',
        'LBFGS Residual': 'deepskyblue'
    }

    # Create the plot
    sns.lineplot(data=data_long, x='Iterations', y='value', hue='algorithm', palette=custom_palette)

    plt.yscale('log')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='upper right', fontsize=15)

    conditioning = "LBFGS/" + folder
    plt.savefig(PLOTS_PATH.format(conditioning, "BFGS-LBFGS-gradient-comparison.png"))
    plt.show()


if __name__ == "__main__":
    # plot all 3D plots
    """
    plot_lambda_epsilon_3D('error')
    plot_lambda_epsilon_3D('time')
    plot_lambda_epsilon_3D('iterations')
    plot_epsilon_memory_3D('error')
    plot_epsilon_memory_3D('iterations')
    """
    plot_norm_gradient_2D("well")
    plot_norm_gradient_2D("ill")
    # plot_awls_vs_exact_gradient_decrease_2D()
    # plot_iteration_memory("well")
    plot_iteration_memory("ill")
    BFGS_comparison()
