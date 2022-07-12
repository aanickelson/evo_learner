"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""
from tqdm import tqdm
import numpy as np
from teaming.domain import DiscreteRoverDomain as Domain
from scipy.stats import sem
from neuralnet import EvolveNN as evoNN


class BasicEvo:
    def __init__(self, env, trial_num):
        self.n_gen = 3000
        self.trial_num = trial_num
        self.env = env
        self.evoNN = evoNN(self.env)

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

    def run_evolution(self):
        attrs = [self.min_score, self.max_score, self.avg_score, self.sterr_score, self.avg_false]
        attr_names = ["min", "max", "avg", "sterr", "false"]
        i = 0
        for gen in tqdm(self.generations):
            scores = []
            falses = []
            candidates = self.evoNN.mutate_weights(self.evoNN.start_weights)

            for c in candidates:
                sc, fs = self.evoNN.score_genome(c)
                scores.append(sc)
                falses.append(fs)

            self.update_logs(scores, falses, i)
            self.evoNN.start_weights = self.evoNN.update_weights(self.evoNN.start_weights, candidates, np.array(scores))

            if i > 0 and not i % 100:
                for j in range(len(attrs)):
                    nm = attr_names[j]
                    att = attrs[j]
                    np.savetxt("trial{}_{}.csv".format(self.trial_num, nm), att, delimiter=",")
            i += 1
        print(f"ending score: {self.evoNN.score_genome(self.evoNN.start_weights)[0]}")
        self.evoNN.score_genome(self.evoNN.start_weights)

        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            np.savetxt("trial{}_{}.csv".format(self.trial_num, nm), att, delimiter=",")


if __name__ == '__main__':
    n_agents = 1
    n_POI = 1
    poi_opt = [[100, 1, 0]]
    with_agents = True
    trial = 1
    env = Domain(n_agents, n_POI, poi_opt, with_agents)
    evo = BasicEvo(env, trial)
    evo.run_evolution()

