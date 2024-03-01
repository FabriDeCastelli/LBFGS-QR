from config import QR_RESULTS, PLOTS_PATH
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def nanoseconds_to_seconds(x, pos):
    return f"{x / 10**9:.2f}"


def plot_lambda(kind):
    if kind not in ['error', 'time']:
        raise ValueError("kind must be one of 'error', 'time'")

    df = pd.read_csv(QR_RESULTS.format("statisticsQR-lambda-m300n20--{}.csv".format(kind)))
    lambda_values = df['lambda']

    if kind == 'time':
        meantime_values = df['meantime']
        stdtime_values = df['stdtime']
        plt.plot(lambda_values, meantime_values, 'o-', label=kind)
        plt.fill_between(
            lambda_values,
            meantime_values - stdtime_values,
            meantime_values + stdtime_values,
            alpha=0.3,
            label='stdtime',
            color='orange'
        )
        plt.title('QR running time with respect to λ')
        plt.ylabel(kind + ' (s)')
        plt.ylim(0, 3.5 * max(meantime_values + stdtime_values))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_seconds))
    else:
        relative_values = df['relative']
        residual_values = df['residual']
        plt.plot(lambda_values, relative_values, 'o-', label='relative')
        plt.plot(lambda_values, residual_values, 'o-', label='residual')
        plt.title('QR error with respect to λ')
        plt.yscale('log')

    plt.xlabel('λ')
    plt.legend()
    plt.xscale('log')

    plt.savefig(PLOTS_PATH.format("QR-lambda-{}.png".format(kind)))
    plt.show()


if __name__ == '__main__':
    plot_lambda('error')
    plot_lambda('time')
