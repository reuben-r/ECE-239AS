import torch as torch 
import torch.nn as nn


import torch 
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size:int, action_size:int, hidden_size:int=256,non_linear:nn.Module=nn.ReLU):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer

        This is a seperate class because it may be useful for the bonus questions
        """
        super(MLP, self).__init__()
        #====== TODO: ======
        # self.linear1 = 
        # self.output = #output layer
        # self.non_linear = non_linear()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #====== TODO: ======
        raise NotImplementedError

class Nature_Paper_Conv(nn.Module):
    """
    A class that defines a neural network with the following architecture:
    - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
    - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
    - 1 convolutional layer with 64 3x3 kernels with a stride of 1x1 w/ ReLU activation
    - 1 fully connected layer with 512 neurons and ReLU activation. 
    Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
    """
    def __init__(self, input_size:tuple[int], action_size:int,**kwargs):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        **kwargs: dict
            additional kwargs to pass for stuff like dropout, etc if you would want to implement it
        """
        super(Nature_Paper_Conv, self).__init__()
        #===== TODO: ======
        # self.CNN = nn.Sequential(*[
        #     
        # ])
        # self.MLP =
        super(Nature_Paper_Conv, self).__init__()

        self.CNN = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the feature map after the convolutional layers
        conv_out_size = self._get_conv_out(input_size)

        self.MLP = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def _get_conv_out(self, shape):
        o = self.CNN(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #==== TODO: ====
        conv_out = self.CNN(x).view(x.size()[0], -1)
        return self.MLP(conv_out)

        
        
    
    


