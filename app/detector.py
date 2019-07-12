import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from breeds import breeds
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from extract_bottleneck_features import *              


class detector():
    def __init__(self, model):
        self.ResNet_model = model
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


    def paths_to_tensor(img_paths):
        '''
        returns a list of tensors from a list of image paths
        '''
        list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
        return np.vstack(list_of_tensors)


    def ResNet50_predict_labels(self, img_path):
        '''
        returns prediction vector for image located at img_path
        '''
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))

    
    def dog_detector(img_path):
        '''
        returns "True" if a dog is detected in the image stored at img_path
        '''
        prediction = ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    
    def face_detector(self, img_path):
        '''
        returns "True" if face is detected in image stored at img_path
        '''
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0


    def ResNet50_predict_breed(self, img_path):
        '''
        Uses the model to predict the breed for the given image
        '''
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = ResNet_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]


    def breed_detector(self, img_path):
        '''
        If a dog is detected in the image, returns the tuple ('Dog', 'predicted breed').
        If a human is detected in the image, returns a tubple the tuple ('Human', 'the resembling dog breed')
        If neither is detected in the image, returns None
        '''
        if dog_detector(img_path):
            breed = ResNet50_predict_breed(img_path)
            return ('Dog', breed)
        if face_detector(img_path):
            breed = ResNet50_predict_breed(img_path)
            return ('Human', breed)
        else:
            None


    def test_breed_detector(self, path):
        '''
        Performs a test of the breed_detector algorithm over the images found in 
        the given path
        '''

        images = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        titles = []

        i = 0
        for img_path in images:
            print('Processing image {}/{}'.format(i + 1, len(images)))
            try:
                prediction = breed_detector(img_path)

                title = ''
                if prediction is not None:
                    title = 'A {} was detected. The predicted/resembling breed is {}'.format(prediction[0], prediction[1])
                else:
                    title = 'Neither a Dog or a Human was detected'

                titles.append(title)
            except:
                print('Unable to open file: {}'.format(img_path))
            i += 1

        nrows = (len(images) + 1) // 2
        ncols = 2

        fig, axs = plt.subplots(nrows, ncols, figsize=(25, 25))
        for k in range(len(images)):
            img = cv2.imread(images[k])
            cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            i = k // 2
            j = k % 2        
            axs[i, j].imshow(cv_rgb)            
            axs[i, j].set_title(titles[k])    

        if len(images) % 2 == 1:
            fig.delaxes(axs[nrows-1][ncols-1])
        
        plt.show()