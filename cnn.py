from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1/255)
x_train=train_datagen.flow_from_directory(r'/Users/Ch Sai Ranjitha/Desktop/chest_xray/chest_xray/train',target_size=(64,64),batch_size=32,class_mode='binary')
x_test=test_datagen.flow_from_directory(r"/Users/Ch Sai Ranjitha/Desktop/chest_xray/chest_xray/test",target_size=(64,64),batch_size=32,class_mode='binary')
print(x_train.class_indices)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=128,init='random_uniform',activation='relu'))
model.add(Dense(output_dim=1,init='random_uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
import tensorflow as tf
tf.compat.v1.global_variables
model.fit_generator(x_train,steps_per_epoch=163,epochs=10,validation_data=x_test,validation_steps=20)
model.save('cnn.h5')
from keras.models  import load_model
from keras.preprocessing import image
import numpy as np
model=load_model("cnn.h5")
img=image.load_img(r"C:\Users\Ch Sai Ranjitha\Desktop/IM-0001-0001.jpeg",target_size=(64,64))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=model.predict_classes(x)
pred
index=['Normal','Pneumonia']