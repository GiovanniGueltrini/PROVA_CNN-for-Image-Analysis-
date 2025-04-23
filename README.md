# Introduzione

This project aims to explore the feasibility of training a Convolutional Neural Network with a very limited quatity of images and verify that it can be applied to medical images.
We will training an CNN with 100 sintetic image and we will apply the GradCAM library to see if the network pay attention to the object of interest.
## Dataset
The dataset is composed of synthetic images generated specifically for training. Each image starts as a random noise background, to which a circular shape is added in a subset of cases. The radius of each circle is randomly selected within a predefined range, simulating a region of interest.

To make the task more realistic and prevent the model from overfitting on simple visual patterns, we also inject additional noise within the circle itself,.


### Model

To create the model I got inspired by the ResNet netwotk's.
The Network is composed by 8 block, each block is made by two block: 
1. Convolution Block: reduces the spatial dimension by convolution with stride 2, and applies dropout and batch normalization.
2. Identity Block: keeps the same size, adding a second convolutional step.

In the end of the network there is a classificator form by two fully connected layer.

#### GradCam
