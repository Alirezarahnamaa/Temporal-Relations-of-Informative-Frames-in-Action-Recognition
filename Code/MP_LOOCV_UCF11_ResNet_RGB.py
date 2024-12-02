import os
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow.keras
from sklearn.utils import shuffle
import csv
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense,GRU
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns

path = '/MVIP/Resnet_UCF11_RGB_featuresByVideo/'
# P is the number of parts for temporal pooling
p=4

# Temporal pooling to extract 4 strong feature vector from spatial features of each frame
def TemporalPooling(videoData):
    data = []

    startIndex = 0
    endIndex = 0
    frameCount = len(videoData)
    PartOfVideo = int(frameCount/p)
    if frameCount < p:
        return
    for _ in range(p):
        endIndex += PartOfVideo
        data.append(np.asarray(np.max(videoData[startIndex:endIndex], axis=0)))
        startIndex += PartOfVideo
    return data

# Load path of train data(features) and labels
fileNames = []
yNames=[]
for root, dirs, files in os.walk(path,topdown=True):
    for name in files:
        if  ('xtrain' in name) and name.endswith((".npz")):
            filePath = root+"/"+name
            fileNames.append(filePath)

        if  ('ytrain' in name) and name.endswith((".npz")):
            filePath = root+"/"+name
            yNames.append(filePath)

# Load train data(features) and labels
filePathIndex = 0
train=[]
Label = []
for filePathIndex in range(len(fileNames)):
    file_class_name = fileNames[filePathIndex].split('/')[-2]
    x = np.load(fileNames[filePathIndex],allow_pickle=True)
    train_data = x['arr_0']
    y = np.load(yNames[filePathIndex],allow_pickle=True)
    train_label = y['arr_0']
    for z in range(len(train_data)):
      if type(train_data[z]) == list :
        train.append(np.squeeze(np.asarray(train_data[z])))
        Label.append(train_label[z])
        continue

      train.append(np.squeeze(train_data[z]))
      Label.append(train_label[z])
    del train_data , x


data=[]
label=[]
data = np.asarray(train)
label = np.asarray(Label)

# Load path of test data(features) and labels
test_fileNames = []
test_yNames=[]
for root, dirs, files in os.walk(path,topdown=True):
    for name in files:
        if  ('xtest' in name) and name.endswith((".npz")):
            filePath = root+"/"+name
            test_fileNames.append(filePath)

        if  ('ytest' in name) and name.endswith((".npz")):
            filePath = root+"/"+name
            test_yNames.append(filePath)

# Load test data(features) and labels

filePathIndex = 0
test=[]
testLabel = []
for filePathIndex in range(len(test_fileNames)):
    file_class_name = test_fileNames[filePathIndex].split('/')[-2]
    x = np.load(test_fileNames[filePathIndex],allow_pickle=True)
    train_data = x['arr_0']
    y = np.load(test_yNames[filePathIndex],allow_pickle=True)
    train_label = y['arr_0']
    for z in range(len(train_data)):
      if type(train_data[z]) == list :
        test.append(np.squeeze(np.asarray(train_data[z])))
        testLabel.append(train_label[z])
        continue

      test.append(np.squeeze(train_data[z]))
      testLabel.append(train_label[z])
    del train_data , x


test_data=[]
test_label=[]
test_data = np.asarray(test)
test_label = np.asarray(testLabel)

# Concatenate all train and test data for Leave-One-Out Cross Validation
inputs = np.concatenate((data, test_data), axis=0)
targets = np.concatenate((label, test_label), axis=0)

# Extract temporal features from spatial features
x_train = []
y_train=[]
for i in range(len(inputs)):
    x=(np.asarray(TemporalPooling(inputs[i])))

    if x.shape != (4, 2048):
      continue
    else:
      x_train.append(x)
      y_train.append(targets[i])
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape)
print(y_train.shape)

inputs , targets = shuffle(x_train, y_train, random_state=0)


# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
CM = []
test_video=[]
Yhat=[]
label=[]
# Model configuration

#batch_size = 32
batch_size = 64
loss_function = sparse_categorical_crossentropy
no_epochs = 80
#optimizer = Adam()
optimizer = 'SGD'
verbosity = 1
num_folds = len(inputs)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model = Sequential()
  model.add(LSTM(40, input_shape=(4, 2048)))
  #model.add(GRU(40, input_shape=(4, 2048)))
  model.add(Dropout(0.2))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout( 0.2))
  model.add(Dense(11 , activation='softmax'))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} of {num_folds}')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)

  y_prediction = model.predict(inputs[test])
  Y_prediction = tensorflow.argmax(y_prediction, axis=1)
  Y_prediction = np.asarray(Y_prediction)
  CM.append(confusion_matrix(targets[test], Y_prediction , normalize='true'))

  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])
  '''
  test_video.append(x_seq[test])
  Yhat.append(Y_prediction)
  label.append(targets[test])
  '''
  test_video.append(test)
  Yhat.append(Y_prediction.item())
  label.append(targets[test].item())
  
  tensorflow.keras.backend.clear_session()
  # Increase fold number
  fold_no = fold_no + 1
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
variance = str(np.std(acc_per_fold))
mean = str(np.mean(acc_per_fold))
meanloss= str(np.mean(loss_per_fold))

# Save results in a CSV file
with open('/storage/users/arahnama/DataPreproc/MVIP/LOOCV_UCF11_RGB_ResNet50/40LSTM_Maxpool_LOOCV_RN_RGB.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow(['accuracy', 'loss', 'vid_seq','Y_hat','Y'])
    
    # Write the data
    for acc, lss, Seq_vid, yhat, y in zip(acc_per_fold, loss_per_fold, test_video, Yhat, label):
        writer.writerow([acc, lss, Seq_vid, yhat, y])
    writer.writerow([mean,meanloss,variance])

#Save trained model
model.save('/MVIP/LSTM_LOOCV_ResNet_RGB.h5')