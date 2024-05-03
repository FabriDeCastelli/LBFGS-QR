# read a csv of two columns and plot them in 2d with seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONDITIONING_RESULTS, PLOTS_PATH


def plot_conditioning():
    df = pd.read_csv(CONDITIONING_RESULTS.format('conditioning.csv'))
    sns.set(style="whitegrid")
    sns.set_context('paper')

    plt.figure(figsize=(11, 8))
    sns.lineplot(data=df, x='lambda', y='condnum', marker='o')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('λ', fontsize=20)
    plt.ylabel('k(X^)', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')

    plt.title('Condition number of X^ with respect to λ', fontsize=25, pad=20, fontweight='bold')

    plt.savefig(PLOTS_PATH.format('conditioning', 'conditioning.png'), bbox_inches='tight', dpi=100)
    plt.show()


if __name__ == '__main__':
    plot_conditioning()
