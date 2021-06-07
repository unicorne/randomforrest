# codingchallenge-cornelius

The goal of this case is to implement a simple classifier for the MNIST dataset (https://deepai.org/dataset/mnist) using Python, running in a Docker container.
Use this repository. Please pay attention to a proper commit log along the case.

# Part 1: Docker 
	
1.	Create a “dummy” main.py file, which will be the starting point for the Python app. This file will be filled in Part 2 and 3. 
3.	Create a Dockerfile that can be used to build a Docker image for Python 3.9, that copies all required project files into the image, including a requirements.txt file that is “pip install”-ed during the build of the image. 
4.	When running the Docker image, the main.py file is supposed to be executed as the CMD. 

# Part 2: Python, scikit-Learn

1.	In the main.py file, use the following snippet to download the MNIST-Dataset, where X will contain the Features and y the Labels of the Binary Classification task:  
`
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home="/app/scikit")
`  
2.	Create the train and test split. For the MNIST-Dataset the first 60.000 datasets are the train and the other ones the test slice. 
3.	Train a Random Forest model on the train split 
4.	Use the trained Random Forest model to predict the test split 
5.	Calculate and print the accuracy of the trained model 
6.	Setup a parameter optimization loop to tune the Random Forest model by testing different Random Forest parameterizations and pick the best model afterwards. 

# Part 3: Python, flask (Optional)

Use flask to setup a small REST API. When calling the API for example with localhost:3000/13 the prediction for the dataset at index 13 of the test split should be returned. 

# Part 4: docker-compose (Optional)

Create a docker-compose file that allows to start the Docker image with the simple command: 

docker-compose up
