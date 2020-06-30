import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

dtype = torch.float
# Running on the gpu; to change to cpu, switch "cuda:0" to "cpu"
device = torch.device("cpu")
# Creating a debug variable to prevent unnecssary printing
DEBUG = False

# Setting parameters
num_images = 60000
# Size of one image
input_size = 784
# Size of our custom hidden nodes
hidden_sizes = [128, 64]
# Size of the output vector
output_size = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

# Downloading and loading in the image data from the MNIST website
trainset = datasets.MNIST('./MNIST_Train', download=True, train=True, transform=transform)
valset = datasets.MNIST('./MNIST_Test', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Setting params for the nn training
# Defines how how much the network learns from each iteration
learning_rate = 1e-3
# How many iterations the system goes through
num_iter = 1

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

# Setting up the optimizer
optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)

# Training the network on our input images
for t in range(num_iter):
    iter_counter = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)

        # Reset the gradients
        optimizer.zero_grad()

        # Now pass the input to the model for training
        nn_guess = model(images)
        nn_guess = nn.functional.log_softmax(nn_guess, dim=1)

        # Computing the loss function with built-in negative log-likelihood function
        loss = nn.functional.nll_loss(nn_guess, labels)
        print(nn_guess)
        print(labels)

        # Use autograd to compute the gradients of the weights
        loss.backward()

        if iter_counter % 63 == 0:
            print()
            print("Image Number:")
            print(iter_counter)
            print("Loss:")
            print(loss.item())
            print()

        iter_counter += 1

        # Update the optim's weights
        optimizer.step()