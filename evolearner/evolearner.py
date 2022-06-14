import numpy as np
import heapq as hq
from evo_learner.evolearner.neuralnet import NeuralNetwork as NN
from teaming.domain import DiscreteRoverDomain as Domain


class EvoLearner:
    def __init__(self, n_agents, n_poi, poi_opt):
        self.n_agents = n_agents            # number of agents
        self.n_poi = n_poi                  # number of POI
        self.poi_options = poi_opt          # POI options
        self.pop_size = 20                  # population size for evolution
        self.policies = []                  # policies
        self.rewards = []                   # array of rewards earned for each policy
        self.env = Domain(self.n_agents, self.n_poi, self.poi_options)
        self.n_in = self.env.state_size()   # number of inputs from the state
        self.n_out = self.n_poi + 1         # number of outputs (actions)
        self.reset()

    def reset(self):
        self.policies = [NN(self.n_in, self.n_out) for _ in range(self.pop_size)]
        self.rewards = [[] for _ in range(self.pop_size)]

    def pick_n_best(self, n):
        """
        Parameters
        ----------
        n: number of neural networks to retain

        Returns
        -------
        n best neural networks based on average score
        """
        avgs = []
        for row in self.rewards:
            avgs.append(np.mean(row))                           # get average scores - cannot be done directly through numpy because arrays have different sizes
        dummy_vals = [i for i in range(self.pop_size)]          # add dummy ID in case of ties (otherwise it throws an error)
        zipped = list(zip(avgs, dummy_vals, self.policies))     # zip lists
        hq.heapify(zipped)                                      # create a heap
        return [nns for _, _, nns in hq.nlargest(n, zipped)]    # find n best from heap and return associated NNs

    def evaluate(self, iterations):
        for _ in range(iterations):
            # select n_agents number of agents
            indices = np.random.randint(0, self.pop_size, self.n_agents)
            agents = [self.policies[i] for i in indices]
            # get score for each
            G = self.env.run_sim(agents)
            # add score to rewards
            for ind in indices:
                self.rewards[ind].append(G)

        print(self.rewards)

        # dummy values for testing
        # for i in range(self.pop_size):
        #     self.rewards[i] = np.random.randint(0, 100, 10)

    def evolve(self):


if __name__ == '__main__':
    evo = EvoLearner(n_agents=5, n_poi=30, poi_opt=[[100, 1, 0]])
    evo.evaluate(10)

    print(evo.rewards)
    print(evo.pick_n_best(3))
