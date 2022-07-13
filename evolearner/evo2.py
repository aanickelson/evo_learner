"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""
from tqdm import tqdm
import numpy as np
from teaming.domain import DiscreteRoverDomain as Domain
from scipy.stats import sem
from evo_learner.evolearner.neuralnet import EvolveNN as evoNN


class BasicEvo:
    def __init__(self, env, p):
        self.n_gen = 3000
        self.trial_num = p.trial_num
        self.env = env
        self.evoNN = evoNN(self.env, p)

        self.generations = range(self.n_gen)
        self.min_score = np.zeros(self.n_gen)
        self.max_score = np.zeros(self.n_gen)
        self.avg_score = np.zeros(self.n_gen)
        self.sterr_score = np.zeros(self.n_gen)
        self.avg_false = np.zeros(self.n_gen)

    def update_logs(self, scores, falses, i):
        # scores = [score_genome(c) for c in candidates]
        self.min_score[i] = min(scores)
        self.max_score[i] = max(scores)
        self.avg_score[i] = np.mean(scores)
        self.sterr_score[i] = sem(scores)
        self.avg_false[i] = np.mean(falses)

    def save_data(self):
        attrs = [self.min_score, self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.evoNN.start_weights]
        attr_names = ["min", "max", "avg", "sterr", "false", "weights"]
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            np.savetxt("trial{}_{}.csv".format(self.trial_num, nm), att, delimiter=",")

    def run_evolution(self):
        for gen in tqdm(self.generations):
            scores = []
            falses = []
            candidates = self.evoNN.mutate_weights(self.evoNN.start_weights)

            for c in candidates:
                sc, fs = self.evoNN.score_genome(c)
                scores.append(sc)
                falses.append(fs)

            self.update_logs(scores, falses, gen)
            self.evoNN.start_weights = self.evoNN.update_weights(self.evoNN.start_weights, candidates, np.array(scores))
            if gen > 0 and not gen % 100:
                self.save_data()

        print(f"ending score: {self.evoNN.score_genome(self.evoNN.start_weights)[0]}")
        self.evoNN.score_genome(self.evoNN.start_weights)
        self.save_data()

