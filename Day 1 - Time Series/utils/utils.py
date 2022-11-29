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
# Finally, we load the data using the pandas API for reading in dat files. Python with pandas is used in a wide variety
# of academic and commercial domains, including finance, neuroscience, economics, statistics, advertising, web analytics,
# and more. The pandas library is an open source, BSD-licensed project providing easy-to-use data structures and
# analysis tools for the Python programming language. The pandas library features a fast and efficient DataFrame object
# for data manipulation with integrated indexing as well as tools for reading and writing data between in-memory data
# structures and different formats such as CSV and text files, Microsoft Excel, SQL databases, and the fast HDF5 format.
# Check out the pandas documentation for more info.
#
# Next, split dataset to training and validation datasets
# We need os for access to the file system, The pickle module implements binary protocols for serializing and
# de-serializing a Python object structure, numPy for fast array math, pandas for data management, MatPlotLib for
# visualization, and train_test_split for spliting arrays or matrices into random train and test subsets.
# ======================================================================================================================
# load dataset
def load_org_data_only_process(dataset, expand_flag):
   # read the raw data file
   with open(dataset, "rb") as raw_file:
       raw_data = pickle.load(raw_file)

   # obtain torque and lable
   temp = pd.DataFrame([[item['torque'], item['label']] for item in raw_data], columns=["torque","label"])
   temp_X, temp_y = temp['torque'], temp['label']

   # format the torque and label
   X, y = [], []
   for item in temp_X:
       values = item.values
       X.append(values)
   X = pd.DataFrame(X).fillna(0)

   for item in temp_y:
       values = item[0]
       y.append(values)
   y = pd.DataFrame(y)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

   print(y_train.value_counts())
   print(y_test.value_counts())

   # expand to 3-dim
   if expand_flag == True:
       X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

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


#===============================original data (process + task)==========================================
def load_org_data_process_task(dataset, expand_flag):
   # read the raw data file
   with open(dataset, "rb") as raw_file:
       raw_data = pickle.load(raw_file)

   # obtain torque and lable
   temp = pd.DataFrame([[item['torque'],
                         item['tcp_pose_0'], item['tcp_pose_1'], item['tcp_pose_2'], item['tcp_pose_3'], item['tcp_pose_4'], item['tcp_pose_5'],
                         item['tcp_speed_0'], item['tcp_speed_1'], item['tcp_speed_2'], item['tcp_speed_3'], item['tcp_speed_4'], item['tcp_speed_5'],
                         item['tcp_force_0'], item['tcp_force_1'], item['tcp_force_2'], item['tcp_force_3'], item['tcp_force_4'], item['tcp_force_5'],
                         item['label']] for item in raw_data], columns=["torque", "tcp_pose_0", "tcp_pose_1", "tcp_pose_2", "tcp_pose_3", "tcp_pose_4", "tcp_pose_5",
                                                                        "tcp_speed_0", "tcp_speed_1", "tcp_speed_2", "tcp_speed_3", "tcp_speed_4", "tcp_speed_5",
                                                                        "tcp_force_0", "tcp_force_1", "tcp_force_2", "tcp_force_3", "tcp_force_4", "tcp_force_5", "label"])


   X, y = [], []
   X.append(restructure_data(temp['torque']))
   X.append(restructure_data(temp['tcp_pose_0']))
   X.append(restructure_data(temp['tcp_pose_1']))
   X.append(restructure_data(temp['tcp_pose_2']))
   X.append(restructure_data(temp['tcp_pose_3']))
   X.append(restructure_data(temp['tcp_pose_4']))
   X.append(restructure_data(temp['tcp_pose_5']))
   X.append(restructure_data(temp['tcp_speed_0']))
   X.append(restructure_data(temp['tcp_speed_1']))
   X.append(restructure_data(temp['tcp_speed_2']))
   X.append(restructure_data(temp['tcp_speed_3']))
   X.append(restructure_data(temp['tcp_speed_4']))
   X.append(restructure_data(temp['tcp_speed_5']))
   X.append(restructure_data(temp['tcp_force_0']))
   X.append(restructure_data(temp['tcp_force_1']))
   X.append(restructure_data(temp['tcp_force_2']))
   X.append(restructure_data(temp['tcp_force_3']))
   X.append(restructure_data(temp['tcp_force_4']))
   X.append(restructure_data(temp['tcp_force_5']))

   X = np.dstack(X)
   print(X.shape)

   for item in temp['label']:
       values = item[0]
       y.append(values)
   y = pd.DataFrame(y)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

   print(y_train.value_counts())
   print(y_test.value_counts())

   # # expand to 3-dim
   # if expand_flag == True:
   #     X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

   print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

   return X_train, X_test, y_train, y_test

