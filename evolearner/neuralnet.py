"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies

Structure and main functions for basic
"""

import torch
from torch import nn
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hid_size, out_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(inplace=True),
            nn.Linear(hid_size, out_size),
            nn.ReLU(inplace=True),
        )
        self.model.requires_grad_(False)

    def run(self, x):
        return self.model(x)

    def get_weights(self):
        d = self.model.state_dict()
        return [d['0.weight'], d['2.weight']]

    def set_weights(self, weights):
        d = self.model.state_dict()
        d['0.weight'] = weights[0]
        d['2.weight'] = weights[1]
        self.model.load_state_dict(d)

    def forward(self, x):
        x = torch.Tensor(x)
        flat_x = torch.flatten(x)
        logits = self.model(flat_x)
        return logits


class EvolveNN:
    def __init__(self, env, p):
        self.env = env
        self.sigma = p.sigma
        self.learning_rate = p.learning_rate
        self.model = NeuralNetwork(env.state_size(), p.hid, env.get_action_size())
        self.start_weights = self.model.get_weights()

    def score_genome(self, weights):
        self.model.set_weights(weights)
        self.env.new_env()
        G, avg_false = self.env.run_sim([self.model])
        return G, avg_false

    def mutate_weights(self, weights):
        weights_to_try = []
        for _ in range(100):
            noise = self.sigma*torch.normal(0, 1, size=weights[0].shape)
            noise2 = self.sigma*torch.normal(0, 1, size=weights[1].shape)
            weights_to_try.append([weights[0] + noise, weights[1] + noise2])
        return weights_to_try

    def update_weights(self, start_weights, weights, scores):
        if scores.std() == 0:
            return start_weights
        scores = (scores - scores.mean()) / scores.std()
        new_weights = []
        for index, w in enumerate(start_weights):
            layer_pop = [p[index] for p in weights]
            update_factor = self.learning_rate / (len(scores) * self.sigma)
            nw = 0
            for j, layer in enumerate(layer_pop):
                nw += np.dot(layer, scores[j])
            nw = start_weights[index] + update_factor * nw
            new_weights.append(nw)
        return new_weights

