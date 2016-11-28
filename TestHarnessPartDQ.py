from EpsilonGreedy import *
from OptimisticGreedy import *

def run():

    """
    Step 1: Initialize the environment
    """
    #Initialize bandit
    bandit = Bandit(10,0,1)
    stationary = True

    pulls = 1000
    runs = 2000

    """
    Step 2: Initialize all of the algorithms you wish to test
    """
    algorithms = []
    
    #initialize Epsilon Greedy algorithm
    alpha = 0.1
    epsilon = 0.1
    epsilonGreedy = EpsilonGreedy(bandit, alpha, epsilon)
    #algorithms.append(epsilonGreedy)
    
    #epsilon greedy with different parameters
    alpha = 0.05
    epsilon = 0.4
    epsilonGreedy2 = EpsilonGreedy(bandit, alpha, epsilon)
    #algorithms.append(epsilonGreedy2)
    
    #optimistic greedy
    alpha = 0.1
    initialValues = 5
    optimisticGreedy = OptimisticGreedy(bandit, initialValues, alpha)
    algorithms.append(optimisticGreedy)

    """
    Step 4: Run the tests
    """
    results = testAlgorithms(bandit, algorithms, pulls, runs, stationary)

    """
    Step 5: Analyze the results
    """
    rewardsDict = results['rewards']
    rewards = rewardsDict[optimisticGreedy]

    #print("Rewards: ") 
    #print(rewards)
    optimalActionsDict = results['optimalActions']
    optimalActions = optimalActionsDict[optimisticGreedy]

    averageReward = np.sum(rewards) / pulls
    pctCorrect = np.sum(optimalActions) / pulls

    print("pct correct: " + str(pctCorrect))
    print("avg reward: " + str(averageReward))
    print("Q: " + str(epsilonGreedy.Q))

    fig = plt.figure(1)
    fig.suptitle('Bandit', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    #titleLabel = "Stationary: " + str(isStationary) + ", eps:" + str(eps) + ", alpha:" + str(alpha)
    #titleLabel = "Average Return Over 2000 Bandits with 1000 pulls each"
    #ax.set_title(titleLabel)
    ax.set_xlabel('Initial')
    ax.set_ylabel('Pct Correct')

    ax.plot(optimalActions)
    plt.show()



def testAlgorithms(bandit, algorithms, numberOfPulls, numberOfRuns, isStationary):
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
        if run % 100 == 0:
            print("Executing run " + str(run) + " ... ")
        #print("++++++++++++++ Run ++++++++++++++ ")
        for pull in range(numberOfPulls):
            for algorithm in algorithms:
                #ask algorithm for the arm it should pull
                arm = algorithm.policy()
                #print("Arm: " + str(arm))    
                #pull the arm and collect the reward
                reward = bandit.pull(arm)
                #print("Reward: " + str(reward))
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
        
                #print("Step size: " + str(stepSize))
                #print("Before update: " + str(rewardArray[pull]))
                rewardArray[pull]+=stepSize*(reward - rewardArray[pull])
                #print("After update: " + str(rewardArray[pull]))
                                
                optimalActionArray[pull]+=stepSize*(optimalAction - optimalActionArray[pull])
                    
                #Walk the bandit if necessary
                if (not isStationary):
                    bandit.walk()
            
        #Finished the run, Reset to a new bandit.
        bandit.reset()
        
        #Reset each algorithm (the Q values etc.)
        for algorithm in algorithms:
            algorithm.reset()
                

    #return the dictionaries of learning results as a tuple
    return({'rewards': algorithmRewards, 'optimalActions':algorithmOptimals})

    
"""
# epsilonGreedyLearnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary=True, eps=0.5, alpha=-1):

returnResults = []
results = []

initialValues = [1/4, 1/2, 1, 2, 4]
stationary = True
#stationary = False
numberOfPulls = 1000
numberOfRuns = 2000
startOfAverage = 100000
for i in range(0, len(initialValues)):
    print("Testing value: " + str(initialValues[i]))
    tempResults = tb.optimisticLearnMultipleRuns(numberOfRuns, numberOfPulls, initialValues[i], stationary, eps=0, alpha=0.1) 
    rewardSum = 0
    results = tempResults[0]
    returnResults.append(np.mean(results))
    #for j in range(startOfAverage, len(results)):
     #   rewardSum += results[j]
    #returnResults.append(rewardSum/startOfAverage)
print(returnResults)


fig = plt.figure(1)
fig.suptitle('Bandit', fontsize=14, fontweight='bold')
ax = fig.add_subplot(211)
#titleLabel = "Stationary: " + str(isStationary) + ", eps:" + str(eps) + ", alpha:" + str(alpha)
#titleLabel = "Average Return Over 2000 Bandits with 1000 pulls each"
#ax.set_title(titleLabel)
ax.set_xlabel('Initial')
ax.set_ylabel('Average Reward')
for i in range(0, len(initialValues)):
    ax.plot(initialValues[i], returnResults[i], label="1/n (Stat.)")
"""
"""
plt.show()

"""
