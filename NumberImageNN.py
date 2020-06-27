import torch
import numpy as np

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
# Mapping the image nodes to a value b/t 0.01:1 will make calculations much easier
mapping = 0.99 / 255

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
learning_rate = 1e-2
# How many iterations the system goes through
num_iter = 500

############################################################################################
# Running the training of the neural network
for t in range(num_iter):
    # Array that stores the error for each image as a unique node
    errors = torch.empty(num_images, device=device, dtype=dtype)
    for i in range(len(img_in)):
        # The network makes a guess based on each image and the weights
        # Implements a ReLU with clamp(min=0) so that we zero out any negative results

        # First clones the input image row and transposes it
        hidden_node = img_in[i].clone()
        hidden_node = torch.transpose(hidden_node, -1, 0)

        if DEBUG:
            # Printing out dimensions for the first dot product
            # Expected sizes: weight1 = 100 x 784; img_in[i] = 784 x 1
            print("Weight 1 Dimensions")
            try:
                print(len(weight1), len(weight1[0]))
            except TypeError:
                print(len(weight1), 1)

            print("Hidden Node Dimensions")
            try:
                print(len(hidden_node), len(hidden_node[0]))
            except TypeError:
                print(len(hidden_node), 1)

        # Then calculates the intermediate guess based on weight1
        # The resulting matrix should be of size 100 x 1
        intermediate_guess = torch.matmul(weight1, hidden_node).clamp(min=0)

        if DEBUG:
            # Printing out dimensions for the first dot product
            # Expected sizes: weight2 = 10 x 100; intermediate_guess = 100 x 1
            print("Weight 2 Dimensions")
            try:
                print(len(weight2), len(weight2[0]))
            except TypeError:
                print(len(weight2), 1)

            print("Intermediate Guess Dimensions")
            try:
                print(len(intermediate_guess), len(intermediate_guess[0]))
            except TypeError:
                print(len(intermediate_guess), 1)

        # Then calculates the final neural network guess using the intermediate guess and
        # the second weight
        nn_guess = torch.matmul(weight2, intermediate_guess).clamp(min=0)

        if DEBUG:
            # Printing out dimensions for the final result
            # Expected size: nn_guess = 10 x 1
            print("Neural Network Guess Dimensions")
            try:
                print(len(nn_guess), len(nn_guess[0]))
            except TypeError:
                print(len(nn_guess), 1)

        # Creating a one_hot array to calculate the error and loss
        # The expected result should be an array of all zeros except for a
        # single 1 in the img_label[i]'s digit location
        num_array = [*range(0, 10, 1)]
        for index in range(10):
            num_array[index] = int(num_array[index] == img_label[i])
        correct_result = torch.tensor(num_array, device=device, dtype=dtype, requires_grad=True)

        if DEBUG:
            # Printing out dimensions for the correct_result
            # Expected size: nn_guess = 10 x 1
            print("Correct Result Dimensions")
            try:
                print(len(correct_result), len(correct_result[0]))
            except TypeError:
                print(len(correct_result), 1)

        # Calculating the error based on the nn_guess and the provided label
        errors[i] = (nn_guess - correct_result).pow(2).sum()

        if DEBUG:
            print()
            print("Resulting Error")
            print(errors[i])
    #loop

    # Computing the total loss of the guess
    # Creates a matrix of a single number, representing the total loss
    loss = errors.clone().sum()

    # Prints out the loss every 50 iterations
    if t % 50 == 0:
        print()
        print(t, loss.item())

    print("t, loss.item()")
    print(t, loss.item())

    # Backprop with pyTorch for stochastic gradient descent
    loss.backward(retain_graph=True)

    print("Weight1 and Weight2 grad")
    print(weight1.grad, weight2.grad)

    # Need to wrap with torch.no_grad() so that this operation is not tracked by autograd
    with torch.no_grad():
        # Update our weights based on the backprop generated by our loss function
        weight1 -= learning_rate * weight1.grad
        weight2 -= learning_rate * weight2.grad

        # Reset the gradients after updating the weights
        weight1.grad.zero_()
        weight2.grad.zero_()
        errors = 0
#loop
