import numpy as np
import heapq as hq
from math import ceil
import torch
import copy
from time import time
import csv
from random import seed, shuffle


from evo_learner.evolearner.neuralnet import NeuralNetwork as NN
from teaming.domain import DiscreteRoverDomain as Domain

class EvoLearner:
    def __init__(self, env, tot_epochs=10000):
        self.pop_size = 20                  # population size for evolution
        self.num_keep = 10                  # number to keep at each evolution step
        self.policies = []                  # policies
        self.rewards = []                   # array of rewards earned for each policy
        self.tot_epochs = tot_epochs
        self.avg_rew_per_epoch = np.zeros(self.tot_epochs)
        self.max_rew_per_epoch = np.zeros(self.tot_epochs)
        self.env = env
        self.n_in = self.env.state_size()   # number of inputs from the state
        self.n_out = self.env.get_action_size()      # number of outputs (actions)
        self.hid = ceil(self.n_in + self.n_out / 2)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.avg_falses = []
        self.reset()

    def reset(self, policies=None):
        if not policies:
            self.policies = [NN(self.n_in, self.hid, self.n_out) for _ in range(self.pop_size)]
        else:
            self.policies = copy.deepcopy(policies)
        self.rewards = np.zeros((self.pop_size, 2))

    def pick_n_best(self):
        """
        Parameters
        ----------
        n: number of neural networks to retain

        Returns
        -------
        n best neural networks based on average score
        """
        avgs = self.rewards[:, 0]
        n = self.num_keep
        dummy_vals = [i for i in range(self.pop_size)]          # add dummy ID in case of ties (otherwise it throws an error)
        shuffle(dummy_vals)                                     # Shuffle them so the order is not deterministic
        zipped = list(zip(avgs, dummy_vals, self.policies))     # zip lists
        hq.heapify(zipped)                                      # create a heap
        return [nns for _, _, nns in hq.nlargest(n, zipped)], [dummy for _, dummy, _ in hq.nlargest(n, zipped)]    # find n best from heap and return associated NNs

    def write_csv(self, data):
        with open('{}_basic_evo_one_poi_type}.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data)

    def evaluate(self, iterations):
        avg_falses = []
        for _ in range(iterations):
            # select n_agents number of agents
            indices = np.random.randint(0, self.pop_size, self.env.n_agents)
            agents = [self.policies[i] for i in indices]
            # get score for each
            max_g = self.env.theoretical_max_g

            global_rew, avg_false = self.env.run_sim(agents)
            # G = global_rew / max_g
            G = global_rew
            avg_falses.append(avg_false)
            # add score to rewards
            for ind in indices:
                curr_rew, tot = self.rewards[ind]
                if self.rewards[ind][0] < G:
                    self.rewards[ind][0] = G    # Keep only the best score
                # self.rewards[ind][0] = ((curr_rew * tot) + G) / (tot + 1)        # running average
                self.rewards[ind][1] += 1
            self.env.reset()
        self.avg_falses.append(np.mean(avg_falses))

    def evolve(self, n_best, sigma):
        num_evo = ceil((self.pop_size - self.num_keep) / self.num_keep)           # Number of times to evolve each NN
        new_pop = copy.deepcopy(n_best)

        # Evolve each best network num_evo number of times
        for _ in range(num_evo):
            for parent in n_best:
                # Limit to pop size (in case num_evo * num_best != pop_size)
                if len(new_pop) < self.pop_size:
                    # Copy the parent NN
                    child = copy.deepcopy(parent)
                    # Alter the weights
                    for param in child.parameters():
                        with torch.no_grad():       # this just makes it work. Because gradient torch things
                            param += sigma * torch.from_numpy(np.random.normal(0, 1, param.shape)).type(torch.FloatTensor).to(self.device)  # I took and slightly altered this line from someone else's project. It seems to work.
                    # Add child to new population
                    new_pop.append(child)
        # Reset the population to the new population
        self.reset(policies=new_pop)

    def train(self, trial_num, epochs=50, sigma=0.5):
        start = time()
        self.evaluate(epochs)
        self.avg_rew_per_epoch[0] = np.mean(self.rewards[:, 0])
        self.max_rew_per_epoch[0] = np.max(self.rewards[:, 0])
        for i in range(1, self.tot_epochs):
            self.env.new_env()      # Reset the environment to a new random configuration
            n_best, idxes = self.pick_n_best()
            self.evolve(n_best, sigma)
            self.evaluate(epochs)
            self.avg_rew_per_epoch[i] = np.mean(self.rewards[:, 0])
            self.max_rew_per_epoch[i] = np.max(self.rewards[:, 0])

            if not i % 100:
                np.savetxt("trial{}_avg.csv".format(trial_num), self.avg_rew_per_epoch, delimiter=",")
                np.savetxt("trial{}_max.csv".format(trial_num), self.max_rew_per_epoch, delimiter=",")

                np.savetxt("trial{}_false.csv".format(trial_num), self.avg_falses, delimiter=',')
                if i > 0:
                    print("{:9}: {}".format(i, np.mean(self.avg_rew_per_epoch[i - 100: i])))
                    print("{:9}: {}".format("max", np.mean(self.max_rew_per_epoch[i - 100: i])))

                    print("avg false:", np.mean(self.avg_falses[-100:]))

        np.savetxt("trial{}_avg.csv".format(trial_num), self.avg_rew_per_epoch, delimiter=",")
        np.savetxt("trial{}_max.csv".format(trial_num), self.max_rew_per_epoch, delimiter=",")

        np.savetxt("trial{}_false.csv".format(trial_num), self.avg_falses, delimiter=',')
        print("{:9}: {}".format(self.tot_epochs, np.mean(self.avg_rew_per_epoch[-10:])))
        print("{:9}: {}".format("max", np.max(self.avg_rew_per_epoch[-10:])))
        print("time:", time() - start)
        print("avg false:", np.mean(self.avg_falses[-100:]))


if __name__ == '__main__':
    seed(0)
    n_agents = 1
    n_poi = 1
    trial_num = 8
    poi_opt = [[1, 1, 0]]  # , [10, 1, 1]]
    env = Domain(n_agents, n_poi, poi_opt)

    evo = EvoLearner(env, tot_epochs=10000)
    evo.train(trial_num, sigma=0.5)

