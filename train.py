"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
#import wandb
import random
import numpy as np

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "minigpt"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    config.log_interval = 10000
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
#if config.to_log:
    #wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""
def train_bigram(num_epochs):
    #print("called__")
    model.to(device)
    #print("model to device")
    lossfunc = torch.nn.CrossEntropyLoss()
    #print("loss defined")
    optimizer = torch.optim.Adam(model.parameters())
    #print("opt to device")
    best_loss = float('inf')
    avg_loss = 0.0
    train_losses = []
    validation_losses = []
    #print("preloop")
    for epoch in range(num_epochs):
        #print("here")
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.squeeze(1).to(device)
            #print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            #print(outputs.shape)
            loss = lossfunc(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                train_losses.append(avg_loss)
                print("Running loss: ", running_loss)
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}')
                if config.to_log:
                    #wandb.log({"training_loss": avg_loss})
                    print("Training loss: ", avg_loss)
                running_loss = 0.0

                # validate

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for j, (inputs, targets) in enumerate(eval_dataloader):
                        inputs, targets = inputs.to(device), targets.squeeze(1).to(device)
                        #inputs = inputs.unsqueeze(1)
                        outputs = model(inputs).squeeze(1)
                        vloss = lossfunc(outputs, targets)
                        val_loss += vloss.item()
                        if j > 1000:
                            break

                avg_val_loss = val_loss / 1000
                validation_losses.append(avg_val_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
                val_loss = 0.0
                model.train()

            if i > 80100:
                break
            #endloop

        print("Training finished.")
    return train_losses, validation_losses

def sample_n(dataloader, n):
    dataset = dataloader.dataset

    indices = random.sample(range(len(dataset)), n)

    subset = torch.utils.data.Subset(dataset, indices)

    subset_dataloader = DataLoader(subset, batch_size=dataloader.batch_size,
                                   num_workers=dataloader.num_workers)

    return subset_dataloader

def train_minigpt(num_iter):
    model.to(device)
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    validation_losses = []
    validation_mark = num_iter // 10
    validation_size = 250
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.squeeze(1).view(-1).to(device)
        #print(targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        outputs = torch.nn.Flatten(0, 1)(outputs).to(device)
        #print(outputs.shape)
        loss = lossfunc(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            train_losses.append(avg_loss)
            print("Running loss: ", running_loss)
            print(
                f'Step [{i + 1}/{len(train_dataloader)}, Loss: {avg_loss:.4f}')
            if config.to_log:
                print("Training loss: ", avg_loss)
            running_loss = 0.0

        if (i + 1) % validation_mark == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for j, (vinputs, vtargets) in enumerate(eval_dataloader):
                    vinputs, vtargets = vinputs.to(device), vtargets.squeeze(1).view(-1).to(device)
                    voutputs = model(inputs).squeeze(1)
                    voutputs = torch.nn.Flatten(0, 1)(voutputs).to(device)
                    vloss = lossfunc(voutputs, vtargets)
                    val_loss += vloss.item()
                    if j > validation_size:
                        break

            avg_val_loss = val_loss / validation_size
            validation_losses.append(avg_val_loss)
            print(f'Validation Loss: {avg_val_loss:.4f}')
            val_loss = 0.0
            model.train()

        if i > num_iter:
            break


    return train_losses, validation_losses