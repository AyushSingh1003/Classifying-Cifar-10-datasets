Today, I will explain the code for building and evaluating a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. 

Download the dataset from here : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

This code includes loading the dataset, building the model, training it, testing it with a single image, and analyzing results using a confusion matrix."

Step 1: Importing Libraries
The first step is importing the required libraries. 
I use TensorFlow for creating and training the CNN. The datasets, layers, and models modules from TensorFlow help with dataset loading, building layers, and assembling the model. Additionally, matplotlib.pyplot is used to visualize training results and display images.

Step 2: Loading and Normalizing the Dataset
Loading the Dataset: Here, I load the CIFAR-10 dataset using datasets.cifar10.load_data(). CIFAR-10 is a popular dataset consisting of 60,000 color images across 10 classes like airplanes, cats, and cars. It is divided into 50,000 training and 10,000 testing images.

Normalizing the Data: Next, I normalize the pixel values by dividing them by 255.0, converting the range from 0–255 to 0–1. This helps the model train faster and achieve better results.

Verifying the Data Shape: I print the shape of the training and test data to confirm the dimensions. CIFAR-10 images are 32x32 pixels with three color channels (RGB). The training data shape is (50000, 32, 32, 3) and the test data shape is (10000, 32, 32, 3).


Step 4: Compiling and Training the Model
Compiling: I compile the model using the Adam optimizer, sparse categorical crossentropy loss (suitable for multi-class classification), and accuracy as the metric.

Training: I train the model for 10 epochs using model.fit() and validate it on the test data. The training process adjusts the weights to minimize the loss function.


Step 5: Evaluating the Model
After training, I evaluate the model on the test set to calculate its accuracy. For example, an accuracy of 80% means the model correctly predicts 80% of the test images.

Step 6: Visualizing Training Results
Purpose: To understand the model's performance over time, I plot the training and validation accuracy for each epoch.

Plotting: Using matplotlib, I create a line plot showing how accuracy changes across epochs. Ideally, the training and validation curves should converge.


Step 7: Testing a Single Image
Process: I select a test image, reshape it for the model, and predict its class. I compare the predicted label with the true label and display the image for visual confirmation.

Step 8: Confusion Matrix
Purpose: To analyze the model's performance on each class, I compute and display a confusion matrix.

Steps: I predict the classes for all test images, calculate the confusion matrix, and visualize it using ConfusionMatrixDisplay. The diagonal values show correct predictions, and off-diagonal values indicate misclassifications.


