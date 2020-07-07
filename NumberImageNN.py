import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import transforms, datasets

dtype = torch.float
# Running on the gpu; to change to cpu, switch "cuda:0" to "cpu"
device = torch.device("cuda:0")
# Creating a debug variable to prevent unnecssary printing
DEBUG = False
torch.autograd.set_detect_anomaly(True)

# We need to transform the image nodes to tensors and then convert to standard normal distribution
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

# Getting the data from the MNIST pyTorch database and putting the files in the project directory
trainset = datasets.MNIST('./MNIST_Train', download=True, train=True, transform=transform)
valset = datasets.MNIST('./MNIST_Test', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)

# Setting params for the weights
H1, H2, Num_In_Nodes, Num_Out_Nodes = 100, 50, (28*28), 10

# Setting params for the nn training
# Defines how how much the network learns from each iteration
learning_rate = 1e-2

# The number of iterations the system goes through
num_iter = 1
iter_counter = 0

# Simple model with the following layers
# 1. Input Layer - 784 nodes
# 2. Hidden Layer 1 - 100 Nodes (Linear translation & ReLU)
# 3. Hidden Layer 2 - 50 Nodes (Linear translation & ReLU)
# 4. Output Layer - 10 Nodes (Linear translation) [represents the likelihood of each digit 0-9]
# --Ideally, there should be a LogSoftMax function at the end (with a dimension of 1), but I can't get that working yet
model = nn.Sequential(
    nn.Linear(Num_In_Nodes, H1).cuda(),
    nn.ReLU().cuda(),
    nn.Linear(H1, H2).cuda(),
    nn.ReLU().cuda(),
    nn.Linear(H2, Num_Out_Nodes).cuda(),
    nn.LogSoftmax(dim=1).cuda()
)

# Using built-in pyTorch optimizer that automatically updates weights based on the calculated gradients with optimizer.step()
# Specifically using a stochastic gradient descent b/c it's the only back-prop method I know
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer.zero_grad()

############################################################################################
# Training the neural network
# Essentially, it runs through all 60000 training images and guesses a digit for each image
# Then it calculates the loss (how far off) the guess was from the provided label
# In minibatches of 64 images, it updates the weights so that it progressively gets closer and closer to the correct guess

for t in range(num_iter):
    print("Iteration ", t)
    iter_counter = 0

    # Separating the dataset into batches of 64 and putting it into the data loader so we can easily iterate through
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    for image_batch, label_batch in trainloader:


        # First flatten the images to 1x784 arrays
        image_batch = image_batch.view(image_batch.shape[0], -1).cuda()

        # Resetting the optimizer's gradient for a new image batch
        optimizer.zero_grad()

        # Runs the image through the model described above
        nn_guess = model(image_batch).cuda()

        # Calculating negative log-likelihood loss of the most recent batch
        loss = nn.functional.nll_loss(nn_guess, label_batch.cuda()).cuda()
        # Backprop to calculate gradients for each of the weights
        loss.backward()
        # Then use the optimizer to update the weight arrays based on the calculated gradients
        optimizer.step()

        if DEBUG:
            print()
            print("Resulting Error")
            print(iter_counter, loss)
            print()

        # Prints out the loss every 50 iterations
        if iter_counter % 100 == 0:
            print()
            print("Iter: ", iter_counter)
            print("Loss: ", loss.item())
        iter_counter += 1
    #loop
#loop


############################################################################################
# Testing the neural network

num_correct, total_num, temp, guess = 0, 0, 0, 0

for test_image, test_label in valloader:

    # Then flatten the image to the same size as out training images (1, 784)
    test_image = test_image[0].view(1, Num_In_Nodes)

    # Run each test image through the model to generate a log probability spectrum for each digit
    with torch.no_grad():
        log_prob = model(test_image.cuda())

    # Transform the model's guess to a value between 0 and 1, with numbers closer to 1 being confident guesses
    test_guess = torch.exp(log_prob)

    # We use numpy here to creat a list of the 10 digit likelihoods so we can find the max
    test_guess = list(test_guess.cpu().numpy()[0])

    if DEBUG:
        print("TestGuess:", test_guess)
        print()

    # Iterate through the test_guess likelihoods to find the max (the network's guess)
    for index in range(0, 10):
        # This inits temp at the beginning of each test_guess
        if index == 0:
            guess = index
            temp = test_guess[index]
        # Checks if each succeeding index is greater than the previous max
        elif test_guess[index] > temp:
            guess = index
            temp = test_guess[index]

    # print("Guess: ", guess, "     Label: ", test_label)
    if guess == test_label.item():
        num_correct += 1
    total_num += 1
#loop
print()
print("##############################")
print("END RESULTS")
print("Total Tested Images: ", total_num)
print("Total Correct Guesses: ", num_correct)
print("Percentage Correct: ", (num_correct/total_num)*100,"%")
print("##############################")
