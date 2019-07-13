import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from breeds import breeds
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from extract_bottleneck_features import *              
from keras.applications.resnet50 import ResNet50


class detector():
    def __init__(self, model):
        self.custom_model = model
        # define ResNet50 model
        self.ResNet50_model = ResNet50(weights='imagenet')
        self.dog_names = breeds
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


    def path_to_tensor(self, img_path):
        '''
        loads RGB image as PIL.Image.Image type
        convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        '''
        img = image.load_img(img_path, target_size=(224, 224))        
        x = image.img_to_array(img)        
        return np.expand_dims(x, axis=0)


    def ResNet50_predict_labels(self, img_path):
        '''
        returns prediction vector for image located at img_path
        '''
        tensor = self.path_to_tensor(img_path)
        img = preprocess_input(tensor)        
        prediction = self.ResNet50_model.predict(img)
        return np.argmax(prediction)

    
    def dog_detector(self, img_path):
        '''
        returns "True" if a dog is detected in the image stored at img_path
        '''
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    
    def face_detector(self, img_path):
        '''
        returns "True" if face is detected in image stored at img_path
        '''
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0


    def ResNet50_predict_breed(self, img_path):
        '''
        Uses the model to predict the breed for the given image
        '''
        tensor = self.path_to_tensor(img_path)
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(tensor)
        # obtain predicted vector
        predicted_vector = self.custom_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]


    def breed_detector(self, img_path):
        '''
        If a dog is detected in the image, returns the tuple ('Dog', 'predicted breed').
        If a human is detected in the image, returns a tubple the tuple ('Human', 'the resembling dog breed')
        If neither is detected in the image, returns None
        '''
        if self.dog_detector(img_path):
            breed = self.ResNet50_predict_breed(img_path)
            return ('Dog', breed)
        if self.face_detector(img_path):
            breed = self.ResNet50_predict_breed(img_path)
            return ('Human', breed)
        else:
            None