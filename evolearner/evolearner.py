import numpy as np
import heapq as hq
from math import ceil
import torch
import copy
from time import time
import csv

from evo_learner.evolearner.neuralnet import NeuralNetwork as NN
from teaming.domain import DiscreteRoverDomain as Domain

class EvoLearner:
    def __init__(self, n_agents, n_poi, poi_opt):
        self.n_agents = n_agents            # number of agents
        self.n_poi = n_poi                  # number of POI
        self.poi_options = poi_opt          # POI options
        self.pop_size = 20                  # population size for evolution
        self.num_keep = 5                   # number to keep at each evolution step
        self.policies = []                  # policies
        self.rewards = []                   # array of rewards earned for each policy
        self.tot_epochs = 100000
        self.avg_rew_per_epoch = np.zeros(self.tot_epochs)
        self.env = Domain(self.n_agents, self.n_poi, self.poi_options)
        self.n_in = self.env.state_size()   # number of inputs from the state
        self.n_out = self.env.get_action_size()      # number of outputs (actions)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.reset()

    def reset(self, policies=None):
        if not policies:
            self.policies = [NN(self.n_in, self.n_out) for _ in range(self.pop_size)]
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
        zipped = list(zip(avgs, dummy_vals, self.policies))     # zip lists
        hq.heapify(zipped)                                      # create a heap
        return [nns for _, _, nns in hq.nlargest(n, zipped)]    # find n best from heap and return associated NNs

    def write_csv(self, data):
        with open('{}_basic_evo_one_poi_type}.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data)

    def evaluate(self, iterations):
        for _ in range(iterations):
            # select n_agents number of agents
            indices = np.random.randint(0, self.pop_size, self.n_agents)
            agents = [self.policies[i] for i in indices]
            # get score for each
            G = self.env.run_sim(agents)
            # add score to rewards
            for ind in indices:
                curr_rew, tot = self.rewards[ind]
                self.rewards[ind][0] = ((curr_rew * tot) + G) / (tot + 1)        # running average
                self.rewards[ind][1] += 1
            self.env.reset()

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

    def train(self, epochs=20, sigma=0.1):
        start = time()
        for i in range(self.tot_epochs):
            if i > 0:
                n_best = self.pick_n_best()
                self.evolve(n_best, sigma)
            self.evaluate(epochs)
            self.avg_rew_per_epoch[i] = np.mean(self.rewards[:, 0])
            if not i % 100:
                print(np.mean(self.rewards[:, 0]))

        np.savetxt("trial1.csv", self.avg_rew_per_epoch, delimiter=",")
        print("{:8}: {}".format(self.tot_epochs, np.mean(self.avg_rew_per_epoch[-10:])))
        print("time:", time() - start)



if __name__ == '__main__':
    evo = EvoLearner(n_agents=3, n_poi=10, poi_opt=[[100, 1, 0]])
    evo.train()
    # evo.evaluate(20)
    # evo.evolve(evo.pick_n_best(), sigma=0.1)
    #
    # print(evo.rewards)
    # for arr in evo.rewards:
    #     print(np.mean(arr))
    # print(evo.rewards)
    # print(evo.pick_n_best(3))
