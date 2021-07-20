# -*- coding: utf-8 -*-
"""

# Task 1
"""

# Commented out IPython magic to ensure Python compatibility.
#import all libraries we will use for task 1
# %pylab inline --no-import-all
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score

#import the dataset
with np.load('training-dataset.npz') as data:
    img = data['x']
    lbl = data['y']
del data
print(img.shape)
print(type(img))
print(lbl.shape)

#split datasets to training, validation and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(img, lbl, test_size = 0.3, random_state = 666)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 666)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

#normalization, image pixel is between 0-255, according to the formula, we divide by 255 and then reshape
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

x_train = X_train.reshape((-1, 28, 28, 1))
x_val = X_val.reshape((-1, 28, 28, 1))
x_test = X_test.reshape((-1, 28, 28, 1))
print(x_train.shape)

#dummy coding labels
onehot = LabelBinarizer()
d_y_train = onehot.fit_transform(y_train) 
d_y_val = onehot.transform(y_val)
d_y_test = onehot.transform(y_test)
print(d_y_train.shape)

"""## Model 1: KNN """

#tune hyperparameter K using 3-fold cross validation
k_range = range(3, 8)
cv_scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors = k, n_jobs = -1)
  scores = cross_val_score(knn, X_train, d_y_train, cv = 3) #3 folds
  cv_score = np.mean(scores)
  print('k = {}, accuracy on validation set = {:.3f}'.format(k, cv_score))
  cv_scores.append((k, cv_score))

#selecting k that gave the best accuracy on validation set
best_k = max(cv_scores, key = lambda x:x[-1])
print('best_k: {} with validation accuracy of {}'.format(best_k[0], best_k))

#using best parameter k to train KNN on training data 
knn_model = KNeighborsClassifier(n_neighbors = best_k[0], n_jobs = -1) 
knn_model.fit(X_train, d_y_train) 
 
#evaluate fitted KNN on test data 
knn_y_pred = knn_model.predict(X_test)
test_accuracy = accuracy_score(d_y_test, knn_y_pred)
print(test_accuracy)

"""## Model 2: CNN  

"""

#tune hyperparameters, here we tune batch size, dropout rate and learning rate
settings = []
 
for batch in [64, 100, 128]:
  for drop in [0, 0.5, 0.8]:
    for lr in [0.1, 0.01, 0.001]:

      print("batch :", batch)
      print("drop:", drop)
      print('learning rate:', lr)
  
      model = Sequential() 
      #convolution 1 
      model.add(Convolution2D(input_shape=(28,28,1), 
                              filters=32, 
                              kernel_size=5, 
                              strides=1, 
                              activation='relu')) 
      
      #pooling 1 
      model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same')) 
      
      #convolution 2 
      model.add(Convolution2D(filters=64, 
                              kernel_size=5, 
                              strides=1,   
                              activation='relu' 
                              )) 
      
      #pooling 2
      model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))
      
      #convolution 3
      model.add(Convolution2D(filters=128, 
                              kernel_size=5, 
                              strides=1,   
                              activation='relu' 
                              )) 
      
      #pooling 3  
      model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))

      #convolution 4 
      model.add(Convolution2D(filters=256, 
                              kernel_size=5, 
                              strides=1,   
                              activation='relu' 
                              )) 
      
      #pooling 4 
      model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))

      #Flatten, transfer to vectors 
      model.add(Flatten()) 
      
      #Dropout 
      model.add(Dropout(drop))

      #fully connected network 1 
      model.add(Dense(500, activation='relu')) 
      
      #fully connected network 2, 26 because 26 different letters in total
      model.add(Dense(26, activation='softmax')) 

      #earlystopping to prevent overfitting
      early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')

      #reducing learning rate
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    factor = 0.1, 
                                    patience = 1, 
                                    verbose = 1, 
                                    mode = 'min', 
                                    min_delta =0.0001, 
                                    cooldown=0, 
                                    min_lr=0)

      callback_lists = [early_stopping, reduce_lr]

      adam = Adam(lr = lr) 
      
      
      model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=['accuracy'])
      model.fit(x_train,
                d_y_train,
                batch_size = batch,  
                epochs = 5,  
                verbose = 1,
                validation_data = (x_val, d_y_val),
                shuffle = True,  
                callbacks = callback_lists)
    
      loss, acc = model.evaluate(x_val, d_y_val)
      settings.append((batch, drop, lr, acc))

#print best accuracy 
best_accuracy = max(settings, key = lambda x:x[-1])
print(best_accuracy) #lr = 0.001
best_batch, best_drop, best_lr = best_accuracy[:-1]
print(best_batch, best_drop, best_lr)

#using tuned parameters to train model
model = Sequential()

#convolution 1, activation
model.add(Convolution2D(input_shape=(28,28,1),
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='same',
                        activation='relu'))

#pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=1,padding='same'))

#convolution 2, activation
model.add(Convolution2D(filters=64,
                        kernel_size=5,
                        strides=1,
                        padding='same',
                        activation='relu'
                        ))

#pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))

#convolution 3, activation
model.add(Convolution2D(filters=128,
                        kernel_size=5,
                        strides=1,
                        padding='same',
                        activation='relu'
                        ))

#pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))


#convolution 4, activation
model.add(Convolution2D(filters=256,
                        kernel_size=5,
                        strides=1,
                        padding='same',
                        activation='relu'
                        ))

#pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))

#Flatten, transfer to vectors
model.add(Flatten())

#Dropout 
model.add(Dropout(best_drop)) 

#fully connected network 1
model.add(Dense(500,activation='relu'))

#fully connected network 2
model.add(Dense(26, activation='softmax'))

#early stopping, to prevent overfitting
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')

#reducing learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor = 0.1, 
                              patience = 1, 
                              verbose = 1, 
                              mode = 'min', 
                              min_delta =0.0001, 
                              cooldown=0, 
                              min_lr=0)


callback_lists = [early_stopping, reduce_lr]

#optimizer
adam = Adam(lr = best_lr)


model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=['accuracy'])


#training
history = model.fit(x_train,d_y_train,
                    batch_size = best_batch,
                    epochs = 12, 
                    validation_data = (x_val, d_y_val), 
                    verbose = 1, 
                    shuffle = True, 
                    callbacks = callback_lists)

#plotting loss and accuracy of training and validation sets
#accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#save our model
model.save('my_model1.h5')

#make predictions of testing sets and see how accurate those predictions are
loss,acc = model.evaluate(x_test, d_y_test)
print(loss,acc)

"""We have got around 85% accuray on testing set with KNN model and 95% accuracy on testing set with CNN model. So we decided to use CNN for our task 2.

# Task 2
"""

# Commented out IPython magic to ensure Python compatibility.
# import libraries used for task 2
# %pylab inline --no-import-all
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.utils import plot_model 
from skimage.util import random_noise
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from google.colab import drive

# load testing-dataset
test = np.load('test-dataset.npy')
print(test.shape)

# see what images are like before denoise
plt.imshow(test[-1])
plt.show()

# denoise all images and see what they are like now
from scipy import ndimage
import matplotlib.pyplot as plt
testing_filtered = []
for i in range(len(test)):
    new_image = ndimage.median_filter(test[i], 2)
    testing_filtered.append(ndimage.median_filter(new_image, 3))
plt.imshow(testing_filtered[-1])
plt.show()

#define a function to split the images
 def image_crop(data):
  
  testing_cropped = []

  for i in range(len(data)):
    #threshold each image and find contours
    img = (data[i]).astype('uint8')
    _, threshold = cv2.threshold(img.copy(), 10, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
   
    bboxes = []

    #creating bounding boxes from contours
    for f in range(len(contours)):
      bboxes.append(cv2.boundingRect(contours[f]))
 
    split = []
    to_remove = []
 
    #threshold to remove small w and h bbox, and split those with w >= 28
    for j in range(len(bboxes)):
      if (bboxes[j][2] < 20) and (bboxes[j][3] < 17):
        to_remove.append(bboxes[j])
      if bboxes[j][2] >= 30:
        split.append(j)
 
    #modifying bboxes to get half w and move x to (x + w/2)
    for g in split:
      bboxes[g] = (bboxes[g][0], bboxes[g][1], int(bboxes[g][2]/2), bboxes[g][3])
      modified_bboxes = bboxes[g]
      modified_bboxes = (int(bboxes[g][0]) + int(bboxes[g][2]), int(bboxes[g][1]),
                        int(bboxes[g][2]), int(bboxes[g][3]))
      bboxes.append(modified_bboxes)
 
    #removing bboxes with small w and h
    for b in to_remove:
      bboxes.remove(b)

    #sorting bboxes
    bboxes = sorted(np.array(bboxes), key = lambda x: x[0])

    cut = []
    
    for h in range(len(bboxes)):
      images = img[bboxes[h][1]:bboxes[h][1]+bboxes[h][3],
                  bboxes[h][0]:bboxes[h][0]+bboxes[h][2]]
      if images[0].shape > np.max(3):
        cut.append(images)

    cropped = []

    #reshaping the cut images to be able to use CNN
    for image_split in cut:
      crop = image_split.reshape((image_split.shape[0],image_split.shape[1],1))
      crop = np.array(tf.image.resize_with_crop_or_pad(crop, 28, 28))
      img_cropped = crop.reshape(28,28)
      cropped.append(img_cropped)  

    testing_cropped.append(cropped)
  return np.array(testing_cropped)

testing_cropped = image_crop(testing_filtered)
print(len(testing_cropped)) #10000 images

# let's see an example letter from testing_cropped dataset
plt.imshow(testing_cropped[420][0])
plt.show()

#most of images are separated into 4 letters, but still many are into 3 or 5 letters
l=[]
for i in range(len(testing_cropped)):
    l.append(len(testing_cropped[i]))
plt.hist(l)
plt.show()

#make 5 predictions which have highest probability scores by using our CNN model
block_size = 55
predictions = []
top1 = []
top2 = []
top3 = []
top4 = []
top5 = []
final = []
for i in range(10000): 
  crops_number = (len(testing_cropped[i]))
  for sample in testing_cropped[i]:
    imbw = sample > threshold_local(sample, block_size, method = 'mean')
    imbw1 = remove_small_objects(imbw, 10, connectivity=1)
    roi = imbw1
    roi = roi.reshape((roi.shape[0],roi.shape[1],1))
    roi = tf.image.resize_with_crop_or_pad(roi, 28, 28).numpy()
    image = roi.reshape(28, 28)
    pre = model.predict(image.reshape(-1,28,28,1))
    
    for i in pre: 
      '''i is the probability of each letter, in total 26 probability scores, 
      we select the highest 5 and their index is the predicted label'''
      prob1 = np.argsort(i)[-5] + 1
      top5.append(prob1)
      prob2 = np.argsort(i)[-4] + 1
      top4.append(prob2)
      prob3 = np.argsort(i)[-3] + 1
      top3.append(prob3)
      prob4 = np.argsort(i)[-2] + 1
      top2.append(prob4)
      prob5 = np.argsort(i)[-1] + 1
      top1.append(prob5)
 
  pred = top5 + top4 + top3 + top2 + top1

  pred1 = []
  y_pred = []
  for i in pred:
    i = str(i)
    if len(i) == 1:
      s = i.zfill(2)
      pred1.append(s)
    else:
      pred1.append(i)
  
  for step in range(0, len(pred1), crops_number):
    pred2 = pred1[step:step + crops_number]
    pred3 = ''.join(pred2)
    y_pred.append(pred3)

    
  final1 = y_pred.copy()
  final.append(final1)
  top1.clear()
  top2.clear()
  top3.clear()
  top4.clear()
  top5.clear()
  y_pred.clear()

#print(final)

#take the last image as an example, to see the most probable labels, we got 100% accuracy for this image
print(final[-1][-1])
plt.imshow(testing_filtered[-1])
plt.show()

#take another image that includes 3 letters as an example, we got at least 2 of 3 letters correctly predicted
print(final[18][-1])
plt.imshow(testing_filtered[18])
plt.show()

# save to a csv file 
import csv
with open('Predictions.csv', 'w', newline = '') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(final)
