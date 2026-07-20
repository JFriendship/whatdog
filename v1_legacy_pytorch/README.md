# Whatdog
Whatdog is a computer vision program that classifies images of dogs into dog breeds. Whatdog uses a fine-tuned ResNet-18 architecture to make predictions for the 120 different dogbreeds in the dataset. Whatdog uses FastAPI to provide dog breed predictions on new image data provided by the user.

## Dataset
**Source:** [Dog Breed Dataset I Used](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data) | [Original Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)  
**Details:** 20,580 Images | 120 Categories

## Installation
### Guide for Inference
With docker installed, use the Dockerfile to build the image and create/run a container.
1. `docker build -t whatdog-image .`
2. `docker run -p 8080:80 whatdog-image`
3. Go to the URL in a browser and add `/docs` to test it out.
4. From the route`/api/v1/predict`, you can upload an image of a dog and receive a prediction for the most prominent dog breed.

### Guide for Development
1. Fork or Clone the Repository
2. Create a virtual environment and `pip install -r path/to/requirements.txt`  
    -  Don't forget to set your interpreter to the newly created virtual environment
3. Download the dataset. Create a folder called `Images` and move the image data into the folder.
4. You can load the FastAPI server by typing `fastapi dev` into the terminal when in the project directory or you can build the docker image and run the container.


## Methodology
### Obtaining a model baseline
It might have been a bit rash to jump straight to a ResNet-18 model architecture, but I thought the complexity of the task was too difficult for anything close to a standard CNN to handle.  
I started by trying to overfit the fine-tuned ResNet-18 on the training data and then working backwards to make the model less complex.

### Data Augmentation
Since I was fine-tuning ResNet-18, I normalized the data using the mean and standard deviation of the Imagenet1k dataset since that was what the pretrained ResNet-18 model was trained on.

## Project Status
Whatdog is currently still in development.

## Contributions
Contributions are **NOT** currently being accepted. Whatdog is a personal project for my own development, but thank you for considering contributing.