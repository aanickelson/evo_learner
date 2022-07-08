import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from scipy.stats import sem

np.set_printoptions(precision=3)


def average_every_n(n, data):
    data_len = len(data)
    n_results = ceil(data_len / n)
    avg = np.zeros(n_results)
    sterr = np.zeros(n_results)
    for i in range(n_results):
        curr_range = data[(i*n):(i+1)*n]
        avg[i] = np.mean(curr_range)
        sterr[i] = sem(curr_range)
    return avg, sterr


def plot_it(avgs, sterrs, n):
    x_vals = [n*i for i in range(len(avgs))]
    plt.plot(x_vals, avgs, 'k-')
    plt.fill_between(x_vals, avgs-sterrs, avgs+sterrs)
    plt.xlabel("Epoch")
    plt.ylabel("Percent of time null actions chosen")
    plt.title("Percent null actions chosen")
    plt.savefig('trial10_false.png')


if __name__ == '__main__':
    data = np.loadtxt("trial10_false.csv")
    n = 200
    avgs, sterrs = average_every_n(n, data)
    plot_it(avgs, sterrs, n)
