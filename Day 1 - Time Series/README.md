# Time Series Analysis Assignment
## 1. Description

### 1.1 Modeling Time Series Data with Deep Learning Neural Networks 
The goal of this repo is to build deep neural networks that can classify the screwing process data based on time series, or sequential, data.

A time series is a series of data points, ordered in time. For complex systems, each data point will likely be multivariate, meaning there are multiple variables for each data point. Examples of data sets with multivariate time series data are financial markets, air quality measurement, and health records. In each case, the goal is to predict one of the variable values, such as a stock price, pollutant level, or patient outcome, based on the sequential dependence of past data.

In this lab, you'll build deep neural network models to classify the screwing process from time series data contained in screwing records.

The purpose of building such a model, is to provide an analytic framework that professionals can use to detect the anomaly at any time of interest. Such a solution provides essential feedback to factories when trying to assess the quality of screwing process, or raise early warning signs to flag at-risk anomaly in a production setting.

This project and the Screwing process data are provided by AAU Smart Lab.

### 1.2 Task
You’ll use time-series classification to classify screwing process on screwing data collected from AAU screw cell. Specifically, you'll build a deep learning model that more accurately classify 2000+ data samples. In total, there are four categories.
<img src="https://github.com/AAU-RoboticsAutomationGroup/PhDAppliedAIMecManEng/blob/main/Day%201%20-%20Time%20Series/images/ScrewingCell.png" width="1000"/>

## 2. Getting Started
This section shows you how to set up your local programming environment.
### 2.1 Set Up Programming Environment

#### 2.1.1 Install Python 3

Require python 3.9 above. Please refer to [here](https://www.python.org/downloads/) for download and installation. 

#### 2.1.2 Install IDE (optional)

You may also install the IDE for development and testing your script. For example, you
may use [Pycharm](https://www.jetbrains.com/pycharm/) or [VScode](https://code.visualstudio.com/download). Both of 
them support Windows, Mac and Linux. The Day One lecture uses Pycharm.

#### 2.1.2 Install Anaconda (optional)
Anaconda offers the easiest way to perform Python/R data science and machine learning 
on a single machine. Start working with thousands of open-source packages and libraries 
today. 

Please refer to [here](https://www.anaconda.com/#) for download and installation. 

#### 2.1.3 Install Required package
Please use the below commands to clone the repo and install required package.

**A.** You can download the repository, or clone the repository using the following code.
```
git clone https://github.com/AAU-RoboticsAutomationGroup/PhDAppliedAIMecManEng.git
```
**B.** 
Please use the below commands to install required package. The *requirements.txt* is under the Day 1 - 
Time Series folder.
```
pip install -r requirements.txt
```
### 2.2 Structure of this repository

#### 2.2.2 Data folder
You will need to manually create a folder with the name "Data". You will need to download the 
[dataset](https://drive.google.com/file/d/1uBQVp9b_pjIhU7E6EhaDXizVb1PbMgOG/view?usp=share_link) and put it under this folder. 

#### 2.2.3 images folder
It contains the static pictures for this repo. 

#### 2.2.4 loss_acc folder
The training results, including the loss and accuracy, will be plotted and saved as figures in this folder.

#### 2.2.5 models folder
We provide four different deep learning models, *Conv1D*, *ConvLSTM2D*, *LSTM* and *Transformer*, for training and evaluation.

#### 2.2.6 scores folder
The F1, recall, and Precision will be recorded in a json file in this folder.

#### 2.2.7 utils folder
*configure.py* and *utils.py* provide configuration information (e.g., dataset path, paramaters of NN) and functions (e.g., data load), respectively. 

#### 2.2.8 requirements.txt
It specifies the required package.

## Assignment instructions
Several lines are missing in *ConvLSTM2D_org_data.py*, *LSTM_org_data.py* and *TRM_org_data.py*.

You will need to use *Conv1D_org_data.py* as reference and fill the missing code in the above scripts. You need to fine-tune
the parameters of the *configure.py* file.

## Getting help
Ask on Teams or contact Chen Li at email cl@mp.aau.dk.
