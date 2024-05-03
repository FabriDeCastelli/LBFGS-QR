import pandas as pd
from matplotlib import pyplot as plt, ticker
import seaborn as sns
from config import COMPARISON_RESULTS, PLOTS_PATH


def nanoseconds_to_seconds_2f(x, pos):
    return f"{x / 10 ** 9:.2f}"


def nanoseconds_to_seconds_3f(x, pos):
    return f"{x / 10 ** 9:.3f}"


def bytes_to_gib(x, pos):
    return f"{x / 1024 ** 3:.2f}"


def QR_vs_LBFGS_time(conditioning, dimension="n"):
    if conditioning not in ['ill', 'well']:
        raise ValueError("conditioning must be one of 'ill', 'well'")
    if dimension not in ['n', 'm']:
        raise ValueError("dimension must be one of 'n', 'm'")
    if dimension == 'n':
        df = pd.read_csv(COMPARISON_RESULTS.format("QRvsLBFGS-n-m200-" + conditioning + "cond--time.csv"))
    else:
        df = pd.read_csv(COMPARISON_RESULTS.format("QRvsLBFGS-m-n50-" + conditioning + "cond--time.csv"))
    sns.set(style="whitegrid")
    sns.set_context('paper')

    fig, ax1 = plt.subplots(figsize=(11, 8))

    sns.lineplot(data=df, x=dimension, y='meantimeQR', marker='o', label='Mean Time QR', color='blue', ax=ax1)
    ax1.fill_between(
        df[dimension],
        df['meantimeQR'] - df['stdtimeQR'],
        df['meantimeQR'] + df['stdtimeQR'],
        alpha=0.3,
        label='Standard Deviation QR',
        color='lightblue'
    )

    sns.lineplot(data=df, x=dimension, y='meantimeLBFGS', marker='o', label='Mean Time L-BFGS', color='red', ax=ax1)
    ax1.fill_between(
        df[dimension],
        df['meantimeLBFGS'] - df['stdtimeLBFGS'],
        df['meantimeLBFGS'] + df['stdtimeLBFGS'],
        alpha=0.3,
        label='Standard Deviation L-BFGS',
        color='lightsalmon'
    )

    ax1.set_title('QR vs LBFGS time and memory scalability on ' + conditioning + '-conditioned matrix', fontsize=20,
                  pad=20,
                  fontweight='bold')
    ax1.set_xlabel(dimension, fontsize=15, labelpad=20)
    ax1.set_ylabel('time (s)', fontsize=15, labelpad=20)
    ax1.set_ylim(0, None)

    ax1.legend(loc='upper left', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_seconds_2f))

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=dimension, y='meanallocsQR', marker='o', label='GiB Allocation QR', color='green', ax=ax2)
    sns.lineplot(data=df, x=dimension, y='meanallocsLBFGS', marker='o', label='GiB Allocation BFGS', color='purple',
                 ax=ax2)
    ax2.set_ylabel('memory (GiB)', fontsize=15, labelpad=20)
    ax2.set_ylim(0, None)
    ax2.legend(loc='upper right', fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(bytes_to_gib))

    plt.savefig(PLOTS_PATH.format("comparison", "QRvsLBFGS-scalability-time-" + conditioning + f"cond-{dimension}.png"))
    plt.show()


def BFGSvsLBFGS_time(dimension="n"):
    if dimension not in ['n', 'm']:
        raise ValueError("dimension must be one of 'n', 'm'")

    if dimension == 'n':
        df = pd.read_csv(COMPARISON_RESULTS.format("BFGSvsLBFGS-n-m200--time.csv"))
    else:
        df = pd.read_csv(COMPARISON_RESULTS.format("BFGSvsLBFGS-m-n50--time.csv"))
    sns.set(style="whitegrid")
    sns.set_context('paper')

    fig, ax1 = plt.subplots(figsize=(11, 8))

    sns.lineplot(data=df, x=dimension, y='meantimeBFGS', marker='o', label='Mean Time BFGS', color='blue', ax=ax1)
    ax1.fill_between(
        df[dimension],
        df['meantimeBFGS'] - df['stdtimeBFGS'],
        df['meantimeBFGS'] + df['stdtimeBFGS'],
        alpha=0.3,
        label='Standard Deviation BFGS',
        color='lightblue'
    )

    sns.lineplot(data=df, x=dimension, y='meantimeLBFGS', marker='o', label='Mean Time L-BFGS', color='red', ax=ax1)
    ax1.fill_between(
        df[dimension],
        df['meantimeLBFGS'] - df['stdtimeLBFGS'],
        df['meantimeLBFGS'] + df['stdtimeLBFGS'],
        alpha=0.3,
        label='Standard Deviation L-BFGS',
        color='lightsalmon'
    )

    ax1.set_title('BFGS vs L-BFGS time and memory scalability on well-conditioned matrix', fontsize=20, pad=20, fontweight='bold')
    ax1.set_xlabel(dimension, fontsize=15, labelpad=20)
    ax1.set_ylabel('time (s)', fontsize=15, labelpad=20)
    ax1.set_ylim(0, None)

    if dimension == 'n':
        ax1.legend(loc='upper right', fontsize=15)
    else:
        ax1.legend(loc='upper left', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_seconds_3f))

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=dimension, y='meanallocsBFGS', marker='o', label='Bytes Allocation BFGS', color='green',
                 ax=ax2)
    sns.lineplot(data=df, x=dimension, y='meanallocsLBFGS', marker='o', label='Bytes Allocation L-BFGS', color='purple',
                 ax=ax2)
    ax2.set_ylabel('memory (B)', fontsize=15, labelpad=20)
    ax2.set_ylim(0, None)
    ax2.legend(loc='lower right', fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(bytes_to_gib))

    plt.savefig(PLOTS_PATH.format("comparison", f"BFGSvsLBFGS-time-{dimension}.png"))
    plt.show()


def quasi_newton_time(conditioning="well", dogleg=False):
    df = pd.read_csv(COMPARISON_RESULTS.format(f"Quasi-Newton-Comparison-time-{conditioning}cond.csv"))
    sns.set(style="whitegrid")
    sns.set_context('paper')

    plt.figure(figsize=(11, 8))

    sns.lineplot(data=df, x='m', y='meantimeLBFGS', marker='o', label='L-BFGS', color='blue')
    plt.fill_between(
        df['m'],
        df['meantimeLBFGS'] - df['stdtimeLBFGS'],
        df['meantimeLBFGS'] + df['stdtimeLBFGS'],
        alpha=0.3,
        label='Standard Deviation L-BFGS',
        color='lightblue'
    )

    sns.lineplot(data=df, x='m', y='meantimeBFGS', marker='o', label='BFGS', color='red')
    plt.fill_between(
        df['m'],
        df['meantimeBFGS'] - df['stdtimeBFGS'],
        df['meantimeBFGS'] + df['stdtimeBFGS'],
        alpha=0.3,
        label='Standard Deviation BFGS',
        color='lightsalmon'
    )

    sns.lineplot(data=df, x='m', y='meantimeDFP', marker='o', label='DFP', color='purple')
    plt.fill_between(
        df['m'],
        df['meantimeDFP'] - df['stdtimeDFP'],
        df['meantimeDFP'] + df['stdtimeDFP'],
        alpha=0.3,
        label='Standard Deviation DFP',
        color='violet'
    )

    sns.lineplot(data=df, x='m', y='meantimeSR1', marker='o', label='SR1', color='violet')
    plt.fill_between(
        df['m'],
        df['meantimeSR1'] - df['stdtimeSR1'],
        df['meantimeSR1'] + df['stdtimeSR1'],
        alpha=0.3,
        label='Standard Deviation SR1',
        color='lightsalmon'
    )

    if dogleg:
        sns.lineplot(data=df, x='m', y='meantimeDFPDogleg', marker='o', label='DFP with Dogleg', color='orange')
        plt.fill_between(
            df['m'],
            df['meantimeDFPDogleg'] - df['stdtimeDFPDogleg'],
            df['meantimeDFPDogleg'] + df['stdtimeDFPDogleg'],
            alpha=0.3,
            label='Standard Deviation DFP with Dogleg',
            color='lightsalmon'
        )
        sns.lineplot(data=df, x='m', y='meantimeBFGSDogleg', marker='o', label='BFGS with Dogleg', color='green')
        plt.fill_between(
            df['m'],
            df['meantimeBFGSDogleg'] - df['stdtimeBFGSDogleg'],
            df['meantimeBFGSDogleg'] + df['stdtimeBFGSDogleg'],
            alpha=0.3,
            label='Standard Deviation BFGS with Dogleg',
            color='lightgreen'
        )

    plt.title(f'Quasi-Newton methods running time on {conditioning}-conditioned matrix', fontsize=20, pad=20,
              fontweight='bold')
    plt.xlabel('m', fontsize=15, labelpad=20)
    plt.ylabel('time (s)', fontsize=15, labelpad=20)
    plt.ylim(0, None)

    handles, labels = plt.gca().get_legend_handles_labels()

    keyword = 'Standard Deviation'

    filtered_handles = [handle for label, handle in zip(labels, handles) if keyword not in label]
    filtered_labels = [label for label in labels if keyword not in label]

    plt.legend(filtered_handles, filtered_labels, fontsize=15, loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(nanoseconds_to_seconds_2f))

    if dogleg:
        plt.savefig(PLOTS_PATH.format("comparison", f"Quasi-Newton-Comparison-time-{conditioning}cond-Dogleg.png"))
    else:
        plt.savefig(PLOTS_PATH.format("comparison", f"Quasi-Newton-Comparison-time-{conditioning}cond.png"))
    plt.show()


def quasi_newton_memory(conditioning="well"):
    df = pd.read_csv(COMPARISON_RESULTS.format(f"Quasi-Newton-Comparison-time-{conditioning}cond.csv"))
    sns.set(style="whitegrid")
    sns.set_context('paper')

    plt.figure(figsize=(11, 8))

    sns.lineplot(data=df, x='m', y='meanallocsLBFGS', marker='o', label='L-BFGS', color='blue')
    sns.lineplot(data=df, x='m', y='meanallocsBFGS', marker='o', label='BFGS', color='red')
    sns.lineplot(data=df, x='m', y='meanallocsBFGSDogleg', marker='o', label='BFGS with Dogleg', color='green')
    sns.lineplot(data=df, x='m', y='meanallocsDFP', marker='o', label='DFP', color='purple')
    sns.lineplot(data=df, x='m', y='meanallocsDFPDogleg', marker='o', label='DFP with Dogleg', color='orange')
    sns.lineplot(data=df, x='m', y='meanallocsSR1', marker='o', label='SR1', color='violet')

    plt.title(f'Quasi-Newton methods memory allocation on {conditioning}-conditioned matrix', fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('m', fontsize=15, labelpad=20)
    plt.ylabel('bytes (GiB)', fontsize=15, labelpad=20)
    plt.ylim(-1, None)

    plt.legend(loc='upper left', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(bytes_to_gib))

    plt.savefig(PLOTS_PATH.format("comparison", f"Quasi-Newton-Comparison-memory-{conditioning}cond.png"))
    plt.show()


if __name__ == '__main__':
    QR_vs_LBFGS_time("well", "m")
    QR_vs_LBFGS_time("well", "n")
    QR_vs_LBFGS_time("ill", "m")
    QR_vs_LBFGS_time("ill", "n")
    BFGSvsLBFGS_time("n")
    BFGSvsLBFGS_time("m")
    quasi_newton_time("ill")
    quasi_newton_time("well", dogleg=True)
    quasi_newton_time("well", dogleg=False)
    quasi_newton_memory("ill")
    quasi_newton_memory("well")
