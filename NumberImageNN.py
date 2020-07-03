import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

dtype = torch.float
# Running on the gpu; to change to cpu, switch "cuda:0" to "cpu"
device = torch.device("cuda:0")
# Creating a debug variable to prevent unnecssary printing
DEBUG = False
torch.autograd.set_detect_anomaly(True)

# Getting the data from the MNIST database
datapath = "MNIST/"
train_data = np.loadtxt(datapath+"mnist_train.csv", delimiter=",")
test_data = np.loadtxt(datapath+"mnist_test.csv", delimiter=",")
num_images = 60000
# Mapping the image nodes to a value b/t 0.01:1 will make calculations much easier
mapping = 0.99 / 255

# Creating matrices that hold the images
# Each image is stored in a row with the first column holding the correct label
img_in = torch.tensor(train_data[:, 1:], device=device, dtype=dtype) * mapping + 0.01
img_label = torch.tensor(train_data[:, :1], device=device, dtype=dtype)

# Setting params for the weights
H1, H2, Num_In_Nodes, Num_Out_Nodes, Batch_Size = 100, 50, (28*28), 10, 64

# Setting params for the nn training
# Defines how how much the network learns from each iteration
learning_rate = 1e-2
# How many iterations the system goes through
num_iter = 1
# Creating matrices to store the results of the nn_guess and labels
guess_set = torch.zeros(Batch_Size, Num_Out_Nodes, device=device, dtype=dtype)
label_set = torch.zeros(Batch_Size, device=device, dtype=torch.int64)
batch_index = 0

model = nn.Sequential(
    nn.Linear(Num_In_Nodes, H1).cuda(),
    nn.ReLU().cuda(),
    nn.Linear(H1, H2).cuda(),
    nn.ReLU().cuda(),
    nn.Linear(H2, Num_Out_Nodes).cuda()
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

loss = torch.tensor(0, device=device, dtype=dtype)

############################################################################################
# Running the training of the neural network
for t in range(num_iter):
    for i in range(len(img_in)):
        optimizer.zero_grad()

        nn_guess = model(img_in[i])

        # Storing the guess and correct result into matrices for loss computation
        guess_set[batch_index] = nn_guess
        label_set[batch_index] = img_label[i]

        # Calculating loss and updating weights after each minibatch
        if batch_index == 63:
            if DEBUG:
                print(guess_set)
                print(label_set.t())

            # Calculating negative log-likelihood loss of the most recent batch
            loss = torch.tensor(nn.functional.nll_loss(guess_set, label_set.t()), device=device, dtype=dtype, requires_grad=True)
            loss.backward()
            optimizer.step()

            if DEBUG:
                print()
                print("Resulting Error")
                print(i, loss)
                print()

        else:
            batch_index = batch_index + 1

        # Prints out the loss every 50 iterations
        if i % 1000 == 0:
            print()
            print("Iter:")
            print(i)
            print("NN Guess:")
            print(nn_guess)
            print("Label")
            print(img_label[i])
            print()
#loop
