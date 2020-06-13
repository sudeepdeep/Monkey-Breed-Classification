#importing modules
import tensorflow
import os
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D,Dense,Activation
from keras.models import Model
img_size = 224 #taking the size of the image
#creating the neural networks using the MobileNet library because for this calssification it would be so complicated for building all the layers that classify 10 different types of monkeys
#for the sake of simplicity we use  the mobilenet library to generate neural layers automatically
mobile_net = MobileNet(weights = 'imagenet' , include_top = False , input_shape = (img_size,img_size,3))
#we have to train the layers which are not trained
for layer in mobile_net.layers:

    layer.trainable = False
#printing the generated layers by using MobileNet library
for (i,layer) in enumerate(mobile_net.layers):

    print(str(i),layer.__class__.__name__,layer.trainable )
#creating a function
def complete_model(bottom_model,no_of_classes): 
    top_model = bottom_model.output #bottom_model is nothing but our mobilenet layers

    top_model = GlobalAveragePooling2D()(top_model) #creating the top model by our own neural layers

    top_model = Dense(1024,activation = 'relu')(top_model)

    top_model = Dense(1024,activation='relu')(top_model)

    top_model = Dense(512,activation='relu')(top_model)

    top_model = Dense(no_of_classes,activation='softmax')(top_model)

    return  top_model

fc = complete_model(mobile_net,10)

model = Model(inputs = mobile_net.input , outputs = fc) #using the Model models in place of Sequential

print(model.summary()) #printing the summary of the layers

#showing the path to the train and validation images

train_dir = r'C:\Users\dell\Desktop\datasets\monkey breed\training\training'

validation_dir = r'C:\Users\dell\Desktop\datasets\monkey breed\validation\validation'
#ImageDataGenerator is used to generate multiple images from the same images
train_data = ImageDataGenerator(rescale = 1./255,
                                rotation_range= 30,
                                height_shift_range= 0.3,
                                width_shift_range= 0.3,
                                horizontal_flip= True,
                                fill_mode= 'nearest')
#but for the validation of images we no need to generate multiple images from one images
validation_data = ImageDataGenerator(rescale = 1./255)
#flowing from the directories of the train and validation folders by below lines of code
train_data_gen = train_data.flow_from_directory(train_dir,target_size = (img_size,img_size),batch_size = 35, class_mode = 'categorical')

validation_data_gen = validation_data.flow_from_directory(validation_dir,target_size = (img_size,img_size),batch_size = 35,class_mode = 'categorical')
#declaring optimizers and call backs
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
#modelcheckpoint is used to save the model which is having high accuracy and less validation loss
modelcheckpoint = ModelCheckpoint(r'C:\Users\dell\Desktop\projects\monke_breed_classification\vgg.h5',
                                save_best_only= True,
                                verbose = 1,
                                monitor= 'val_loss',
                                mode= 'min' )
#earlystopping is used to stop the training process if the accuracy is not increaing
#patience = 10 means if the accuracy is not increasing it will wait upto 10 epochs and then terminate
early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta= 0,
                                patience=10,
                                verbose=1,
                                restore_best_weights=True)
#reduce learining rate is used to reduce the learning rate is validation loss is not decreasing
reduce_lr  =ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience= 5,
                            verbose=1,
                            min_lr=0.0001)
callbacks = [modelcheckpoint,early_stopping,reduce_lr]
#compiling the model
model.compile(loss= 'categorical_crossentropy',optimizer=Adam(lr = 0.001),metrics = ['accuracy'])

no_of_train_samples = 1098 #count of training images
no_of_valid_samples = 272  #count of validation images
epochs = 25
batch_size = 35

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= no_of_train_samples//batch_size, #dividing with batch size for taking some images at a time to avoid the burden on cpu
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_data_gen,
    validation_steps=no_of_valid_samples//batch_size)
