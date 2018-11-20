# Code the bandit problem

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 200
BANDIT_PROBABILITIES = [0.1,0.3,0.5,0.7,0.9]

class Bandit:
    def __init__(self, p):
        # Similar to a slot machine
        self.p = p
        # a,b are parameters for beta distribution, uniformly distributed
        self.a = 1
        self.b = 1

    def pull(self):
        # Random is between 0,1... if Random < P -> 1, otherwise 0
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x

    def __str__(self):
        return "p:{}, a:{}, b:{}".format(self.p, self.a, self.b)

def plot(bandits, trial):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x,y,label="Real p: {}".format(b.p))
    plt.title('Bandit Dist after {} trials'.format(trial))
    plt.legend()
    plt.show()

def run_experiment():
    # Initialize Bandits with probabilites
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [199]
    for i in range(NUM_TRIALS):
        bestb = None
        maxsample = -1
        allsamples = []
        for b in bandits:
            sample = b.sample()
            allsamples.append(sample)
            if sample > maxsample:
                maxsample = sample
                bestb = b
        if i in sample_points:
            print("Current Samples ", allsamples)
            plot(bandits,i)

        print(bestb)
        x = bestb.pull()
        bestb.update(x)

if __name__ == '__main__':
    run_experiment()
