#importing necessary libraries
import tensorflow
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
model = load_model(r'C:\Users\dell\Desktop\projects\monke_breed_classification\vgg.h5') #loading our model

class_labels = [
	'mantled_howler', 
	'patas_monkey', 
	'bald_uakari', 
	'japanese_macaque', 
	'pygmy_marmoset', 
	'white_headed_capuchin',
	'silvery_marmoset',
	'common_squirrel_monkey', 
	'black_headed_night_monkey',
	'nilgiri_langur' 
	]

def check(path):

    img = image.load_img(path,target_size = (224,224)) #loading our testing image which is preciously not seen by our model

    x = image.img_to_array(img) #converting our loaded image into array

    x = x.astype('float32')/255 #scaling our image

    x = np.expand_dims(x,axis = 0) #expanding the dimensions

    pred = np.argmax(model.predict(x))#predicting that image by using our model

    print('the given monkey is {}'.format(class_labels[pred]))

check(r'C:\Users\dell\Desktop\datasets\monkey breed\test\filename.jpg')

