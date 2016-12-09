#Reinforcement learning for Non-Stationary bandits

##Problem Description
Imagine a slot machine with multiple arms that can be pulled. However, unlike most slot machines, imagine this machine actually has a "best" arm to pull. If the reward you received from each arm was deterministic, learning which arm to pull would be simple. Just pull each arm, keep track of which arm received the most reward, and then start picking that arm each pull in the future. However, for practical bandit problems, it is not so simple for several reasons:
- The values returned are stochastic. So the task of determining which arm to pull requires sampling from several different arms several times
- You must balance exploration with exploitation given your task is to maximize rewards - not just determine which is the "best" arm.
- The mean value of these arms may change over time. This is considered "non-stationary." Given this type of environment, exploration is constantly needed.

##Solution
We will compare several different Reinforcement learning algorithms- and parameters to determine which perform best - primarily in non-stationary environments. These algorithms attempt to balance exploration and exploitation in different ways. Each algorithm can be tweaked appropriately by changing some parameter.

##Usage
There are several important files main files:

1. [TestHarness.py](TestHarness.py)
  * This contains several functions making it easy to test various different algorithms.
2. [Arm.py](Arm.py)
  * Represents one arm within a bandit. 
  * Each arm has a mean and variance from which rewards are stochastically returned
3.  [Bandit.py](Bandit.py)
  * Contains several arms, each of which can be "pulled" to receive a reward.
4. [EpsilonGreedy.py](EpsilonGreedy.py), [Gradient.py](Gradient.py), [OptimisticGreedy.py](OptimisticGreedy.py) [UCB.py](UCB.py)
  * The various learning algorithms.

You can call the learning methods directly, but in addition, there are a few basic ways to test various algorithms, both from within TestHarness.py

```python
from TestHarness import *
"""
It is important to average rewards across several different runs since bandits 
are created randomly. Therefore the results could be incredibly skewed (for 
better or worse) depending on what type of bandit was created. ie. if one arm 
is significantly better than all others, all algorithms will perform well. 
"""
#Plot the average reward at each pull, averaged over 1000 different runs
testUCB(runs = 1000, pulls = 20000, stationary = False, c=2, alpha=0.1)

#Same as above, but for a stationary bandit and using a gradient method
testGradient(runs=1000, pulls=20000, stationary=True, alpha=0.1):

```

It is one thing to test algorithms independently, but you can also test different algorithms and parameter values at the same time using the same bandit. This is important since, if you're comparing algorithms against each other, you want each to have an equal chance against each other. To do so, you can manipulate def testAllAlgorithms() method such that it instantiates the algorithms that you want tested. 


##Results
The following are several results obtained when comparing algorithms against stationary and non stationary bandits. As you can tell, some algorithms continue to perform well (epsilon greedy with a constant step size, UCB). While others (epsilon greedy with average returns, optimistic greedy, and gradient start to perform more poorly as the bandit randomly walks. 

Each graph below was taken as the average over 500 runs of 10,000 steps. Where not explicitly noted, an alpha and epsilon of 0.1 were used.

Blue lines indicate the stationary bandit while green indicates the non-stationary version.

###Epsilon Greedy with using a average return.
 
```python
compareEpsilonGreedy(runs=500, pulls=10000, alpha=-1, epsilon=0.1) #alpha = -1 indicates average return
````
![alt text](Results/EpsilonGreedyStationaryvsNonStationaryAverageReward10000Steps500Runs.png "Epsilon Greedy Compared")
###Epsilon Greedy with a constant step size of 0.1
```python
compareEpsilonGreedy(runs=500, pulls=10000, alpha=0.1, epsilon=0.1)
````
![alt text](Results/EpsilonGreedyStationaryVsNonStationaryConstantStep.png "Epsilon Greedy Compared")
###Upper confidence bound with c = 2 
```python
compareUCB(runs=500, pulls=10000, c=2, alpha=0.1)
````
![alt text](Results/UCBNonStationaryVsStationary.png "UCB")
###Gradient method  
```python
compareGradient(runs=500, pulls=10000, alpha=0.1)
````
![alt text](Results/GradientNonStationaryvsStationary.png "Gradient")

###Optimistic greedy
```python
compareOptimistic(runs=5000, pulls=10000, alpha=0.1, initialValues=5)
````
![alt text](Results/OptimisticGreedyStationaryvsNonStationary.png "Optimistic")

###Baseline Results for Stationary
In order to make sure the algorithms were working, baseline tests were performed and compared against the results in "An Introduction To Reinforcement Learning"
###Epsilon Greedy with a, eps = 0.1
```python
testEpsilonGreedy(runs=2000, pulls=1000, stationary=True, alpha=0.1, epsilon=0.1):
````
![alt text](Results/Stationary/EpsilonGreedy.png "EpsilonGreedy")

###Optimistic with Q0 = 5
```python
testOptimistic(runs=2000, pulls=1000, stationary=True, alpha=0.1, initialValues=5):
````
![alt text](Results/Stationary/Optimistic.png "Optimistic")

###UCB with c = 2
```python
testUCB(2000,1000,True, 2, 0.1)
````
![alt text](Results/Stationary/UCBRewards.png "UCB")

###Gradient with alpha = 0.1
```python
testGradient(2000, 1000, False, 0.1)
````
![alt text](Results/Stationary/Gradient.png "Gradient")

##A sudden change in the environment
The non-stationary environments were designed to change very gradually over time. They wouuld take a random walk of 0,01 every pull. Another result which is interesting is how these algorithms perform if the change isn't gradual but sudden. Which algorithms recover quickly? Which get stuck choosing a sub optimal arm.
To determine this the dynamics were changed that every 1000'th step the arms would be completely shuffled. So the "best" arm would be changed every 1000 steps.
The results are as follows (the blue linne indicates the algorithms performance when the environment is stationary. The green line indicates the algorithms performance when the environment shuffles every 1000 steps:

Epsilon Greedy with weighted average (1/n step size)
![alt text](Results/ShuffleEvery1000/EpsilonAverageReturn.png "Epsilon Average")

Epsilon Greedy with constant step size alpha = 0.1
![alt text](Results/ShuffleEvery1000/EpsilonConstantAlpha.png "Epsilon Constant")

Optimistic, alpha = 0.1
![alt text](Results/ShuffleEvery1000/OptimisticGreedy.png "Optimistic")

Gradient, alpha = 0.1
![alt text](Results/ShuffleEvery1000/Gradient.png "Gradient")

UCB, c=2
![alt text](Results/ShuffleEvery1000/UCB.png "UCB")


##Further Study
