import random 
import torch 
import numpy as np

class ReplayBufferDQN:
    def __init__(self, buffer_size:int, seed:int = 42):
        self.buffer_size = buffer_size
        self.seed = seed
        self.buffer = []
        random.seed(self.seed)
    
    def add(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray
            , done:bool):
        """Add a new experience to the buffer

        Args:
            state (np.ndarray): the current state of shape [n_c,h,w]
            action (int): the action taken
            reward (float): the reward received
            next_state (np.ndarray): the next state of shape [n_c,h,w]
            done (bool): whether the episode is done
        """
        #TODO: Implement the add method
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
        
        
    
    def sample(self,batch_size:int, device = 'cuda'):
        """Sample a batch of experiences from the buffer

        Args:
            batch_size (int): the number of samples to take

        Returns:
            states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
            actions (torch.Tensor): a np.ndarray of shape [batch_size] of dtype int64
            rewards (torch.Tensor): a np.ndarray of shape [batch_size] of dtype float32
            next_states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
            dones (torch.Tensor): a np.ndarray of shape [batch_size] of dtype bool
        """
        samples = random.sample(self.buffer, batch_size)
        #TODO: Implement the sample method
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        return states, actions, rewards, next_states, dones



    def __len__(self):
        return len(self.buffer)
