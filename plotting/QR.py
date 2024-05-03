from config import QR_RESULTS, PLOTS_PATH, nanoseconds_to_seconds, setup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np


def plot_lambda(kind, smoothing_window=2):
    if kind not in ['error', 'time']:
        raise ValueError("kind must be one of 'error', 'time'")

    df = pd.read_csv(QR_RESULTS.format("statisticsQR-lambda-m300n20--{}.csv".format(kind)))
    lambda_values = df['lambda']

    if kind == 'time':

        setup(title='QR running time with respect to λ', y_label='Time (s)', x_label='λ')
        meantime_values = df['meantime']
        stdtime_values = df['stdtime']
        sns.lineplot(data=df, x='lambda', y='meantime', label='Mean Time', color='blue')
        plt.fill_between(
            lambda_values,
            meantime_values - stdtime_values,
            meantime_values + stdtime_values,
            alpha=0.3,
            label='Standard Deviation',
            color='lightblue'
        )
        plt.ylim(0, 3.5 * max(meantime_values + stdtime_values))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_seconds))
    else:
        setup(title='QR errors with respect to λ', y_label='Error', x_label='λ')

        df['relative'] = np.convolve(df['relative'], np.ones(smoothing_window) / smoothing_window, mode='same')
        df['residual'] = np.convolve(df['residual'], np.ones(smoothing_window) / smoothing_window, mode='same')
        df['stability'] = np.convolve(df['stability'], np.ones(smoothing_window) / smoothing_window, mode='same')

        sns.lineplot(data=df, x='lambda', y='relative', label='Relative Error', color='blue')
        sns.lineplot(data=df, x='lambda', y='residual', label='Residual', color='red')
        sns.lineplot(data=df, x='lambda', y='stability', label='Stability', color='green')
        plt.yscale('log')

    plt.legend(loc='upper right', fontsize=15)
    plt.xscale('log', base=10)

    plt.savefig(PLOTS_PATH.format('QR', "QR-lambda-{}.png".format(kind)))
    plt.show()


def plot_forward_error(smoothing_window=5):
    df = pd.read_csv(QR_RESULTS.format("statisticsQR-forward-m300n20--error.csv"))
    setup(title='QR forward error with respect to λ', y_label='Error', x_label='λ')
    df['forwardQ'] = np.convolve(df['forwardQ'], np.ones(smoothing_window) / smoothing_window, mode='same')
    df['forwardR'] = np.convolve(df['forwardR'], np.ones(smoothing_window) / smoothing_window, mode='same')
    sns.lineplot(data=df, x='lambda', y='forwardQ', label='Forward Error on Q', color='blue')
    sns.lineplot(data=df, x='lambda', y='forwardR', label='Forward Error on R', color='red')
    plt.yscale('log')
    plt.xscale('log', base=10)
    plt.legend(loc='upper right', fontsize=15)
    plt.savefig(PLOTS_PATH.format('QR', "QR-forward_error.png"))
    plt.show()


if __name__ == '__main__':
    plot_lambda('error')
    plot_lambda('time')
    plot_forward_error()
