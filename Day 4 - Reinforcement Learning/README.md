# Reinforcement Learning Assignment
## 1. Description
### 1.1 Training your first Reinforcement Learning agent
The goal of this repo is to build deep neural networks that we will train to act as our policy (strategy) for a Reinforcement Learning agent. The neural network will map observations to actions.

<img src="https://user-images.githubusercontent.com/10414639/205514351-04ee86d5-38aa-450c-a641-436a5eef7a13.gif" data-canonical-src="https://user-images.githubusercontent.com/10414639/205514351-04ee86d5-38aa-450c-a641-436a5eef7a13.gif" width="250" />

A reinforcement learning agent can use a table, function or neural network to map observations to actions. Depending on the observations different neural network architectures can be used e.g. CNNs for image data.

In this repo, you'll implement a reinforcement learning agent with the python library [skrl](https://skrl.readthedocs.io/en/latest/). You will set up the neural network for our policy. The environment you will be working on is the classic `cartpole` environment in `Gym` from [gymlibrary.dev](https://www.gymlibrary.dev) (formerly OpenAI Gym).

The purpose of training such an agent, is to have it learn how to take the proper actions to balance the pole on the cart. Similar techniques can be used to control other processess for example in industrial machines and processes.

|-|-|
| --- | --- |
| Action Space | Discrete(2) |
| Observation Shape | (4,) |
| Observation High | [4.8 inf 0.42 inf] |
| Observation Low | [-4.8 -inf -0.42 -inf] |
| Import | `gym.make("CartPole-v1")` |

### 1.2 Task
You’ll use `skrl` to set up and train a `Deep Deterministic Policy Gradient (DDPG)` agent to balance a pole in the `cartpole` environment. Specifically, you'll build a deep learning model that maps observations from the environment to relevant actions that the agent must take to maximize its rewards over time.

#### 1.2.1 Action Space
| Num | Action
|---|---|
|0|Push cart to the left|
|1|Push cart to the right|

#### 1.2.2 Observation Space
|Num|Observation|Min|Max|
|---|---|---|---|
|0|Cart Position|-4.8|4.8|
|1|Cart Velocity|-Inf|Inf|
|2|Pole Angle|~ -0.418 rad (-24°)|~ 0.418 rad (24°)|
|3|Pole Angular Velocity|-Inf|Inf|


## 2. Getting started
The assignment will be set up in Google Colab.

## 3. Training and Evaluation
Train a DDPG agent with various hyperparameters. Try with different neural network architectures, change the number of hidden layers and hidden units, try different learning rates, batch sizes etc. Read more here about DDPG and hyperparameters https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html

## 4. Data Visualization
The training can be followed in Tensorboard where the reward from each timestep/episode is plotted and can easily be compared to previous training runs.

## 5. Assignment instructions
A Jupyter Notebook has been prepared in Google Colab. Implement the sections where specified.

## 6. Expected results
The training should yield a similar result as below.

https://user-images.githubusercontent.com/10414639/205459399-4d8afdf3-608d-44b3-b6e9-d1397478f286.mp4

## 7. Getting help
Ask on Moodle or contact 

* Simon Bøgh, sb@mp.aau.dk
* Nestor Arana Arexolaleiba, narana@mondragon.edu
