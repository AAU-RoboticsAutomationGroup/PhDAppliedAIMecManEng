# ======================================================================================================================
# First, import the libraries we need into our Python workspace.
# We need os for access to the file system, The pickle module implements binary protocols for serializing and
# de-serializing a Python object structure, numPy for fast array math, pandas for data management, MatPlotLib for
# visualization, and train_test_split for spliting arrays or matrices into random train and test subsets.
# ======================================================================================================================
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ======================================================================================================================
# we load the data using the pandas API for reading in dat files. Python with pandas is used in a wide variety
# of academic and commercial domains, including finance, neuroscience, economics, statistics, advertising, web analytics,
# and more. The pandas library is an open source, BSD-licensed project providing easy-to-use data structures and
# analysis tools for the Python programming language. The pandas library features a fast and efficient DataFrame object
# for data manipulation with integrated indexing as well as tools for reading and writing data between in-memory data
# structures and different formats such as CSV and text files, Microsoft Excel, SQL databases, and the fast HDF5 format.
# Check out the pandas documentation for more info.
# ======================================================================================================================
def load_org_data_only_process(dataset, expand_flag):
   # read the raw data file
   with open(dataset, "rb") as raw_file:
       raw_data = pickle.load(raw_file)

   # obtain torque and lable
   temp = pd.DataFrame([[item['torque'], item['label'], item['time']] for item in raw_data], columns=["torque","label", "time"])
   temp_X, temp_y, temp_t = temp['torque'], temp['label'], temp['time']

   # ===================================================================================================================
   # Visualization:
   # uncomment the following code if you want to plot the first sample
   # ===================================================================================================================
   # plt.plot(temp_t[0], temp_X[0])
   # plt.show()

   # format the torque and label
   X, y = [], []
   for item in temp_X:
       values = item.values
       X.append(values)
   X = pd.DataFrame(X).fillna(0)
   y = pd.DataFrame(temp_y)

   # Next, split dataset to training and validation datasets, we use 20% as test dataset
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

   # print the number of the training and test data samples
   print(y_train.value_counts())
   print(y_test.value_counts())

   # expand to 3-dim
   if expand_flag == True:
       X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

   # print the sample data shape
   print(X_train.shape)

   return X_train, X_test, y_train, y_test


# plot the accuracy and loss
def plot_loss_acc(history, loss_img, acc_img):
   plt.plot(history.history['acc'])
   plt.plot(history.history['val_acc'])
   plt.title('model accuracy')
   plt.ylabel('accuracy')
   plt.xlabel('epoch')
   plt.legend(['train', 'val'], loc='upper left')
   plt.savefig(acc_img)
   plt.close()

   # plot loss figure
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'val'], loc='upper left')
   plt.savefig(loss_img)
   plt.close()


def restructure_data(feature_data):
   feature = []
   for item in feature_data:
       values = item.values
       feature.append(values)
   feature = pd.DataFrame(feature).fillna(0)

   return feature
