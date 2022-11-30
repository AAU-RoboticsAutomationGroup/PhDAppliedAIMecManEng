# Natural Language Processing (NLP) Assignment
## 1. Description

### 1.1 Instructing Mobile Industrial Robots through a Virtual Assistant 
The goal of this repo is to a virtual assistant that can interact with the MiR robot on the shop floor.

A industrial virtual assistant is an independent agent who provides services to users while interacting with various industrial robots running on the shop floor. A virtual assistant typically operates from a factory but can access the necessary services, such as control a mobile robot to do a delivery task.

In this repo, you'll build deep neural network models through the [wit.ai](https://wit.ai/) to interact with our Little Helper (LH) robot using voice or text.

The purpose of building such a model, is to provide a natural and flexible interaction environment to enable people to use their voice to control the robots. Such a solution provides essential feedback to manufacturers when trying to introduce intelligent human-robot interaction for enhancing their employee's competency. 

The live demo is arranged at AAU Smart Lab.

### 1.2 Task
You will need to design a simple Virtual Assistant which is able to interact through speech recognition or text to control the LH for the chosen tasks.
<img src="https://github.com/AAU-RoboticsAutomationGroup/PhDAppliedAIMecManEng/blob/main/Day%202%20-%20Natural%20Language%20Processing/images/VA.png" width="1000"/>


## 2. Getting Started
This section shows you how to set up your local programming environment.
### 2.1 Set Up Programming Environment

#### 2.1.1 Install Python 3

Require python 3.9 above. Please refer to [here](https://www.python.org/downloads/) for download and installation. 

#### 2.1.2 Install IDE (optional)

You may also install the IDE for development and testing your script. For example, you
may use [Pycharm](https://www.jetbrains.com/pycharm/) or [VScode](https://code.visualstudio.com/download). Both of 
them support Windows, Mac and Linux. The Day One lecture uses Pycharm.

#### 2.1.3 Install Required package
Please use the below commands to clone the repo and install required package.

**A.** You can download the repository, or clone the repository using the following code.
```
git clone https://github.com/AAU-RoboticsAutomationGroup/PhDAppliedAIMecManEng.git
```
**B.** 
Please use the below commands to install required package. The *requirements.txt* is under the Day 2 - 
Natural Language Processing folder.
```
pip install -r requirements.txt
```
### 2.2 Structure of this repository

#### 2.2.1 images folder
It contains the static pictures for this repo. 

#### 2.2.2 Script for controlling LH
*Mir.py* contains functions for controlling MiR. 

#### 2.2.3 Script for speech service
*speech_service.py* contains text to speech and speech to text functions. 

#### 2.2.4 requirements.txt
It specifies the required package.

We provide detailed comments in each script to help you understand the code.

## 2.3. Connect to LH
This section shows you how to set up your local programming environment.

#### 2.3.1 Connect AAU 5G network
In order to connect to our LH, you will need to connect to our AAU 5G network first. The password will be given on the lecture day.

#### 2.3.2 Log in LH
Access the [mir](mir.com) with the Username: Distributor, The password will be given on the lecture day.

## 3. Building Online Dataset and Training your Model
We will provide a live demo during the lecture to show you how to use [wit.ai](https://wit.ai/) service to build your own dialogue corpus and training the model.

More details of how to access your model and extract the core information from the model prediction can be found [here](https://github.com/wit-ai/pywit). 

## 4. Assignment instructions
There are four tasks in total.
- Task 1: Greeting. Your VA should be able to greet to people.
- Task 2: Checking Battery. Your VA should be able to check how much battery level of the LH.
- Task 3: Reporting Position. Your VA should be able to report the current postion of the LH.
- Task 4: Delivery Package. Your VA should be able to control the LH to perform a package delivery task according to the user's oral command.

You are required to complete Task 4 and any other two tasks.

## 5. Getting help
Ask on Teams or contact Chen Li at email cl@mp.aau.dk.
