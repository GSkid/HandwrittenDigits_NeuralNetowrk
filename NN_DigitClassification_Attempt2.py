import torch
import torchvision
import torch.nn as nn
import torch.optim

dtype = torch.float
# Running on the gpu; to change to cpu, switch "cuda:0" to "cpu"
device = torch.device("cuda:0")
# Creating a debug variable to prevent unnecssary printing
DEBUG = False

# Getting the data from the MNIST database
datapath = "MNIST/"
train_data = np.loadtxt(datapath+"mnist_train.csv", delimiter=",")
test_data = np.loadtxt(datapath+"mnist_test.csv", delimiter=",")
num_images = 60000
# Size of one image
input_size = 784
# Size of our custom hidden nodes
hidden_sizes = [128, 64]
# Size of the output vector
output_size = 10

# Creating matrices that hold the images
# Each image is stored in a row with the first column holding the correct label
img_in = torch.tensor(train_data[:, 1:], device=device, dtype=dtype) * mapping + 0.01
img_label = torch.tensor(train_data[:, :1], device=device, dtype=dtype)

# Setting params for the weights
H, Num_In_Nodes, Num_Out_Nodes = 100, (28*28), 10

# Initializing the weights to random values
# Specifically turning on the requires_grad to true so we can backprop
weight1 = torch.randn(H, Num_In_Nodes, device=device, dtype=dtype, requires_grad=True)
weight2 = torch.randn(Num_Out_Nodes, H, device=device, dtype=dtype, requires_grad=True)

# Setting params for the nn training
# Defines how how much the network learns from each iteration
learning_rate = 1e-1
# How many iterations the system goes through
num_iter = 1