import torch
from torch import nn
import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor


# NN model taken from https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, n_inputs),
            nn.ReLU(),
            nn.Linear(n_inputs, n_outputs * 2),
            nn.ReLU(),
            nn.Linear(n_outputs * 2, n_outputs)
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: inputs as a list

        Returns
        -------

        """
        x = torch.Tensor(x)
        # print(x.size())
        flat_x = torch.flatten(x)
        # print(flat_x.size())
        logits = self.linear_relu_stack(flat_x)
        return logits


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork(6, 8).to(device)
    rand_list = [1,2,3,3,2,1]
    ins = torch.Tensor(rand_list)
    print(list(model.parameters()))
    model.perturb_weights()
    print(list(model.parameters()))
    #
    # outs = model(ins).detach().numpy()
    # print(outs)
    # print(np.argmax(outs))
