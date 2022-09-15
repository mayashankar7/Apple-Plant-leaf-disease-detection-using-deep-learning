import numpy as np
import pandas as pd
import os
from re import search
import shutil
import natsort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

DIR=r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images'

train=pd.read_csv(r"C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\labels\train.csv")
test=pd.read_csv(r"C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\labels\test.csv")

print(train.head())
print(test.head())

image1=Image.open(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\Test_0.jpg')
plt.imshow(image1)
plt.show()

#Prepare the Training Data

class_names=train.loc[:,'healthy':].columns
print(class_names)

number=0
train['label']=0
for i in class_names:
    train['label']=train['label'] + train[i] * number
    number=number+1

print(train.head())

print(DIR)

print(natsort.natsorted(os.listdir(DIR)))

def get_label_img(img):
    if search("Train",img):
        img=img.split('.')[0]
        label=train.loc[train['image_id']==img]['label']
        return label


def create_train_data():
    images=natsort.natsorted(os.listdir(DIR))
    for img in tqdm(images):
        label=get_label_img(img)
        path=os.path.join(DIR,img)

        if search("Train", img):
            if (img.split("_")[1].split(".")[0]) and label.item() == 0:
                shutil.copy(path, r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\healthy')

            elif (img.split("_")[1].split(".")[0]) and label.item() == 1:
                shutil.copy(path, r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\multiple_disease')

            elif (img.split("_")[1].split(".")[0]) and label.item() == 2:
                shutil.copy(path, r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\rust')

            elif (img.split("_")[1].split(".")[0]) and label.item() == 3:
                shutil.copy(path, r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\scab')

        elif search("Test", img):
            shutil.copy(path, r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\test')

try:
    shutil.os.mkdir(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train')
    shutil.os.mkdir(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\healthy')
    shutil.os.mkdir(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\multiple_disease')
    shutil.os.mkdir(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\rust')
    shutil.os.mkdir(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\scab')

    shutil.os.mkdir(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\test')
except FileExistsError:
    print("Already exist")

train_dir=create_train_data()

#Data Preprocessing
Train_DIR=r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train'
Categories=['healthy','multiple_disease','rust','scab']
for j in Categories:
    path=os.path.join(Train_DIR,j)
    for img in os.listdir(path):
        old_image=cv2.imread(os.path.join(path,img),cv2.COLOR_BGR2RGB)
        plt.imshow(old_image)
        plt.show()
        break
    break
IMG_SIZE=224
new_image=cv2.resize(old_image,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_image)
plt.show()

#Model Prepration
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D

datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                validation_split=0.2)
train_datagen=datagen.flow_from_directory(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='training')
val_datagen=datagen.flow_from_directory(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='validation')
model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(4,activation='softmax'))

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
print(model.summary())

checkpoint=ModelCheckpoint(r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\models\apple2.h5',
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True,
                          verbose=1)
earlystop=EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       verbose=1,
                       restore_best_weights=True)

callbacks=[checkpoint,earlystop]
model_history=model.fit_generator(train_datagen,validation_data=val_datagen,
                                 epochs=30,
                                 steps_per_epoch=train_datagen.samples//16,
                                 validation_steps=val_datagen.samples//16,
                                 callbacks=callbacks)
acc_train=model_history.history['accuracy']
acc_val=model_history.history['val_accuracy']
epochs=range(1,31)
plt.plot(epochs,acc_train,'g',label='Training Accuracy')
plt.plot(epochs,acc_val,'b',label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

loss_train=model_history.history['loss']
loss_val=model_history.history['val_loss']
epochs=range(1,31)
plt.plot(epochs,loss_train,'g',label='Training Loss')
plt.plot(epochs,loss_val,'b',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Making the Prediction on a Single Image
test_image=r'C:\Users\mayas\Desktop\BTech\SEMESTER 5\PROJECTS\REVIEW 2\DATASET\images\train\rust\Train_23.jpg'
image_result=Image.open(test_image)

from tensorflow.keras.preprocessing import image
test_image=image.load_img(test_image,target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=test_image/255
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
print(np.argmax(result))
Categories=['healthy','multiple_disease','rust','scab']
image_result=plt.imshow(image_result)
plt.title(Categories[np.argmax(result)])
plt.show()