from EpsilonGreedy import *
from OptimisticGreedy import *
from UCB import *
from Gradient import *

"""
Functions to test algorithm individually. 
"""
def testUCB(runs, pulls, stationary, c, alpha):

    #Initialize bandit
    bandit = Bandit(10,0,1)

    #Initialize all of the algorithms you wish to test
    algorithms = []
    
    #Gradient algorithm
    ucb = UCB(bandit, c, alpha)
    algorithms.append(ucb)
        
    #Run the tests
    results = testAlgorithms(bandit, algorithms, runs, pulls, stationary)

    #Analyze results
    plotAlgorithmOptimalActions(results = results, algorithm = ucb, stationary = stationary, c=c)
    
    
def testGradient(runs, pulls, stationary, alpha):

    #Initialize bandit
    bandit = Bandit(10,0,1)

    #INitialize algorithms
    algorithms = []
    
    #Gradient algorithm
    gradient = Gradient(bandit, alpha)
    algorithms.append(gradient)
        
    #Run tests
    results = testAlgorithms(bandit, algorithms, runs, pulls, stationary)

    #Analyze results
    plotAlgorithmOptimalActions(results=results,algorithm=gradient, stationary=stationary,alpha=alpha)

def testEpsilonGreedy(runs, pulls, stationary, alpha, epsilon):

    #Initialize bandit
    bandit = Bandit(10,0,1)

    #Initialize algorithms
    algorithms = []
    
    #Epsilon Greedy algorithm
    epsilonGreedy = EpsilonGreedy(bandit, alpha, epsilon)
    algorithms.append(epsilonGreedy)
        
    #Run tests
    results = testAlgorithms(bandit, algorithms, runs, pulls, stationary)

    #Analyze results
    plotAlgorithmOptimalActions(results=results, algorithm=epsilonGreedy, stationary=stationary, alpha=alpha, eps=epsilon)       
    
def testOptimistic(runs, pulls, stationary, alpha, initialValues):

    #Initialize bandit
    bandit = Bandit(10,0,1)

    #Initialize algorithms
    algorithms = []

    #Optimistic initial
    optimistic = OptimisticGreedy(bandit, initialValues, alpha)
    algorithms.append(optimistic)

    #Run tests
    results = testAlgorithms(bandit, algorithms, runs, pulls, stationary)

    #Analyze results
    plotAlgorithmOptimalActions(results=results, algorithm=optimistic, stationary=stationary, alpha=alpha, initialValues = initialValues)       
    
"""
Function to test several different algorithms using the same bandit
"""
def testAllAlgorithms():

    """
    Step 1: Initialize the environment
    """
    #Initialize bandit
    bandit = Bandit(10,0,1)
    stationary = False

    #TODO - Enter the actual number of pulls and runs we want to do for testing.
    pulls = 100
    runs = 10

    """
    Step 2: Initialize all of the algorithms you wish to test
    """
    algorithms = []
    
    #Epsilon Greedy algorithms

    #TODO - Enter the actial epsilons we want to test
    epsilons = [0.5, 0.1, 0.15, 0.2]
    greedyAlgorithms = []
    alpha = 0.1
    for epsilon in epsilons:
        epsilonGreedy = EpsilonGreedy(bandit, alpha, epsilon)
        algorithms.append(epsilonGreedy)
        greedyAlgorithms.append(epsilonGreedy)
        
    #Optimistic greedy
    #TODO - Enter teh actual initial values we want to test
    initialValues = [1, 2, 3, 4, 5]
    optimisticAlgorithms = []
    alpha = 0.1
    for initialValue in initialValues:
        optimisticGreedy = OptimisticGreedy(bandit, initialValue, alpha)
        algorithms.append(optimisticGreedy)
        optimisticAlgorithms.append(optimisticGreedy)
        
    #UCB
    alpha = 0.1
    #TODO - Enter the actual c values we want to test
    cValues = [1, 2, 3, 4]
    ucbAlgorithms = []
    for c in cValues:
        ucb = UCB(bandit, c, alpha)
        algorithms.append(ucb)
        ucbAlgorithms.append(ucb)
        
    #Gradient
    #TODO - Enter teh actual alpha values we want to test
    alphas = [0.05, 0.1, 0.15, 0.2]
    gradientAlgorithms = []
    for alpha in alphas:
        gradient = Gradient(bandit, alpha)
        algorithms.append(gradient)
        gradientAlgorithms.append(gradient)
    
    """
    Step 4: Run the tests
    """
    results = testAlgorithms(bandit, algorithms, runs, pulls, stationary)

    """
    Step 5: Analyze the results
    """
    rewardsDictionaryOfAllAlgorithms = results['rewards']
    optimalActionsDictionaryOfAllAlgorithms = results['optimalActions']

    for algorithm in greedyAlgorithms:
        rewards = rewardsDictionaryOfAllAlgorithms[algorithm]
        lastRewards = rewards[int(len(rewards) / 2) : len(rewards) -1]
        avgReward = np.sum(lastRewards) / pulls / 2
        print(algorithm.name + ", epsilon: " + str(algorithm.eps) + ", Average Reward: " + str(avgReward))     
        #TODO - this should be plotted as a data point on a plot, connected to the other greedy points
    
    for algorithm in optimisticAlgorithms:
        rewards = rewardsDictionaryOfAllAlgorithms[algorithm]
        lastRewards = rewards[int(len(rewards) / 2) : len(rewards) -1]
        avgReward = np.sum(lastRewards) / pulls / 2
        print(algorithm.name + ", initial Values: " + str(algorithm.initialValues) + ", Average Reward: " + str(avgReward))     
        #TODO - this should be plotted as a data point on a plot, connected to the other optimistic points
        
    for algorithm in ucbAlgorithms:
        rewards = rewardsDictionaryOfAllAlgorithms[algorithm]
        lastRewards = rewards[int(len(rewards) / 2) : len(rewards) -1]
        avgReward = np.sum(lastRewards) / pulls / 2
        print(algorithm.name + ", c: " + str(algorithm.c) + ", Average Reward: " + str(avgReward))     
        #TODO - this should be plotted as a data point on a plot, connected to the other ucb points
        
    for algorithm in gradientAlgorithms:
        rewards = rewardsDictionaryOfAllAlgorithms[algorithm]
        lastRewards = rewards[int(len(rewards) / 2) : len(rewards) -1]
        avgReward = np.sum(lastRewards) / pulls / 2
        print(algorithm.name + ", alpha: " + str(algorithm.alpha) + ", Average Reward: " + str(avgReward))     
        #TODO - this should be plotted as a data point on a plot, connected to the other gradient points
        

"""
Main test harness function that tests all of the algorithms.
Input: 
- bandit: the bandit which is used to generate rewards
- algorithms: an array of the different algorithms to test
- numberOfPulls: number of pulls to learn and generate reward for
- numberOfRuns: the number of different runs (each containing numberOfPulls). Each run generates a new 
                bandit. You need to do this to receive good average results. If you learn under only one bandit, 
                your results may be skewed better or worse, depending on the variance of the bandit arms.
                The more runs, the more this variance is minimized.
- isStationary: Whether the bandit's arms randomly walk after each pull
"""
def testAlgorithms(bandit, algorithms, numberOfRuns, numberOfPulls, isStationary):
    #Set up the structures for storing the algorithms results
    
    #Dictionary: Keys are the algorithm object. Value is an array of rewards for 
    #each pull 0->numberOfPulls, averaged over numberOfRuns
    algorithmRewards = {}
    
    #Dictionary: Keys are the algorithm object. Value is an array of %'s where 
    #optimal action was taken for each pull 0->numberOfPulls, averaged over numberOfRuns
    algorithmOptimals = {}

    for algorithm in algorithms:
        returns = [0.0]*numberOfPulls
        algorithmRewards[algorithm] = returns
    
        optimals = [0.0]*numberOfPulls
        algorithmOptimals[algorithm] = optimals


    for run in range(numberOfRuns):
        for algorithm in algorithms:
            algorithm.reset()        
        if run % 1 == 0:
            print("Executing run " + str(run) + " ... ")
        for pull in range(numberOfPulls):
            for algorithm in algorithms:
                #ask algorithm for the arm it should pull
                arm = algorithm.policy()
                #pull the arm and collect the reward
                reward = bandit.pull(arm)
                #allow the algorithm to learn based on result of the arm
                algorithm.learn(reward, arm)
                
                #update the statistics for the algorithm
                rewardArray = algorithmRewards[algorithm]
                optimalActionArray = algorithmOptimals[algorithm]
                optimalAction = 0
                
                stepSize = 1/(run+1)                
                if (arm == algorithm.bandit.bestArm()):
                    optimalAction = 1
                else:
                    optimalAction = 0
        
                rewardArray[pull]+=stepSize*(reward - rewardArray[pull])
                optimalActionArray[pull]+=stepSize*(optimalAction - optimalActionArray[pull])
                    
                #Walk the bandit if necessary
                if (not isStationary):
                    bandit.walk()
            
        #Finished the run, Reset to a new bandit.
        bandit.reset()

    #return the dictionaries of learning results as a tuple
    return({'rewards': algorithmRewards, 'optimalActions':algorithmOptimals})

def plotAlgorithmOptimalActions(results, algorithm, stationary, alpha=-1, eps=-1, c=-1, initialValues=-1):
    optimalActionsDictionaryOfAllAlgorithms = results['optimalActions']

    optimalActions = optimalActionsDictionaryOfAllAlgorithms[algorithm]
    
    if not alpha == -1:
        alphaString = " Alpha: " + str(alpha) + " "
    else:
        alphaString = ""
        
    if not eps == -1:
        epsString = " Epsilon: " + str(eps) + " "
    else:
        epsString = ""

    if not c == -1:
        cString = " c: " + str(c) + " "
    else:
        cString = ""
    if not initialValues == -1:
        initialValuesString = " Initial Values: " + str(initialValues) + " "
    else:
        initialValuesString = ""

    fig = plt.figure(1)
    fig.suptitle('Bandit', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    titleLabel = "Algorithm name: " + algorithm.name + " Sationary: " + str(stationary) + alphaString + epsString + cString + initialValuesString
    ax.set_title(titleLabel)
    ax.set_xlabel('Step/Pull')
    ax.set_ylabel('Average reward')

    ax.plot(optimalActions)
    plt.show()


def plotAlgorithmRewards(results, algorithm, stationary, alpha=-1, eps=-1, c=-1, initialValues=-1):
    rewardsDictionaryOfAllAlgorithms = results['rewards']

    rewards = rewardsDictionaryOfAllAlgorithms[algorithm]     
   
    if not alpha == -1:
        alphaString = " Alpha: " + str(alpha) + " "
    else:
        alphaString = ""
        
    if not eps == -1:
        epsString = " Epsilon: " + str(eps) + " "
    else:
        epsString = ""

    if not c == -1:
        cString = " c: " + str(c) + " "
    else:
        cString = ""
    if not initialValues == -1:
        initialValuesString = " Initial Values: " + str(initialValues) + " "
    else:
        initialValuesString = ""


    fig = plt.figure(1)
    fig.suptitle('Bandit', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    titleLabel = "Algorithm name: " + algorithm.name + " Sationary: " + str(stationary) + alphaString + epsString + cString + initialValuesString
    ax.set_title(titleLabel)
    ax.set_xlabel('Step/Pull')
    ax.set_ylabel('Pct Correct')

    ax.plot(rewards)
    plt.show()

    