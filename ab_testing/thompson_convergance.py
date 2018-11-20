import numpy as np
import matplotlib.pyplot as plt
from bayesian_bandit import Bandit
# import pandas as pd

def run_experiment(p1,p2,p3,N):

    bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]
    data = np.empty(N)

    for i in range(N):
        b_list = [b.sample() for b in bandits]

        j = np.argmax(b_list)
        x = bandits[j].pull()
        bandits[j].update(x)

        print(b_list)

        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    print(cumulative_average)
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*p1)
    plt.plot(np.ones(N)*p2)
    plt.plot(np.ones(N)*p3)

    plt.ylim((0,1))
    plt.xscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment(0.2,0.25,0.3,1000)
