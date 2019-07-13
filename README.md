# dog_breed_classifier

## Project: Create a Web Application for Dog Identification 

The goal is to classify images of dogs according to their breed by using Convolutional Neural Networks (CNN)

The app will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

## Instructions
1. Clone or download the repository
2. Move into the project's root directory
3. Download and install necessary requirements
	```
		pip install -r requirements.txt
	```
4. Move to the app directory
5. Run
	```
		python app.py
	```
    *Take into accout that the first runnig may take a while as the app downloads some files for the models being used.*
6. Browse to http://0.0.0.0:3001/ or http://localhost:3001/
7. Upload Image files
    *Take into accout that the first file uploading may take a while as the app downloads some files for the models being used.*

## Info about the used models
I use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github. I have downloaded one of these detectors and stored it in the haarcascades directory.

I use a pre-trained ResNet-50 model to detect dogs in images. In order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the model returns a value between 151 and 268 (the categories corresponding to dogs).

Finaly, I use transfer learning to create a CNN that can identify dog breed from images. The model uses the the pre-trained ResNet50 model as a fixed feature extractor.

#### Steps:
* if a dog is detected in the image, return the predicted breed.
* if a human is detected in the image, return the resembling dog breed.
* if neither is detected in the image, provide output that indicates an error.

#### Test accuracy: 82.8947%

### Credits
This app was developed following the guide of this Udacity's nanodegree: https://eu.udacity.com/course/data-scientist-nanodegree--nd025
I've found help to solve execution problems here: https://stackoverflow.com/questions/51231576/tensorflow-keras-expected-global-average-pooling2d-1-input-to-have-shape-1-1
and here: https://github.com/jrosebr1/simple-keras-rest-api/issues/5



[image1]: ./images/Screenshot_1.png "Sample Output"