from Bandit import *
from pylab import *
import matplotlib.pyplot as plt

"""
Usage:
#import BanditLearn
from BanditLearn import *

#Create the BanditLearnAlgorithm / Environment
algorithm = BanditLearn()

#1000 pulls using a stationary bandit, with 10% exploration, and avergae return for step size:
algorithm.learn(1000, True, 0.1)

#1000 pulls using a Non stationary bandit, with 10% exploration, and avergae return for step size:
algorithm.learn(1000, False, 0.1)

#1000 pulls using a Non stationary bandit, with 10% exploration, and constant step size of 0.2
algorithm.learn(1000, False, 0.1, 0.2)

#2000 runs of 1000 pulls using a non stationary bandit, with 10% exploration, and constant step size of 0.2
algorithm.learnMultipleRuns(2000, 1000, False, 0.1, 0.2)

"""


class BanditLearn:
    def __init__(self):
        self.numberOfArms = 10

        self.Q = [0.0]*self.numberOfArms
        self.bandit = Bandit(self.numberOfArms, 0, 1)
        
    def learnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary = True, eps=0.5, alpha = -1):
        avgRewardVector = np.array([0.0]*numberOfPulls)
        optimalActionVector = np.array([0.0]*numberOfPulls)
        for run in range(numberOfRuns):
            print("Executing run " + str(run))
            learnResults = self.learn(numberOfPulls, isStationary, eps, alpha)
            avgRewardVector+=np.array(learnResults[0])
            optimalActionVector+=np.array(learnResults[1])
        avgRewardVector = avgRewardVector/numberOfRuns
        optimalActionVector = optimalActionVector/numberOfRuns
        
        
        fig = plt.figure()
        fig.suptitle('Bandit', fontsize = 14, fontweight = 'bold')
        ax = fig.add_subplot(111)

        titleLabel = "Stationary: " + str(isStationary) + ", eps:" + str(eps) + ", alpha:" + str(alpha)
        ax.set_title(titleLabel)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Average reward')
        ax.plot(optimalActionVector)
        plt.show()

        return (avgRewardVector, optimalActionVector)
            
    """
    Returns 2 arrays.
    - The first array is the rewards received at each time step
    - The second array contains the optimal action selection. 
    -- 0 if non optimal
    -- 1 if optimal action
    
    In this way, learn can be called multiple times on multiple bandit test beds to receive
    the average of returns at each time step as well as the average optimal 
    """
    def learn(self, numberOfPulls, isStationary = True, eps=0.5, alpha = -1):
    
        self.reset()

        cumulativeReward = 0
        averageRewardArray = []
        rewardArray = []
        cumulativeOptimalAction = 0
        optimalActionPctArray = []
        optimalActionArray=[]
    
        numberOfPullsArray = [0]*self.numberOfArms
        for pull in range(numberOfPulls):

            #Pick an action/arm to pull
            armIndx = 0
            #Decide to explore vs. Exploit
            randomE = random()                
            if (randomE < eps):
                #explore
                armIndex = randint(0,self.numberOfArms)    
            else:
                #Exploit / Choose the best current action
                armIndex= argmax(self.Q)
            
            #Pull the lever
            reward = self.bandit.arms[armIndex].pull()
        
            #incriment the count of number of times it was used (in case we're using average)
            numberOfPullsArray[armIndex]+=1        
        
            #update statistics
            cumulativeReward+=reward

            if (armIndex == self.bandit.bestArm()):
                cumulativeOptimalAction+=1
        
            #Update the Action values
            if (alpha==-1):
                stepSize = 1/numberOfPullsArray[armIndex]
            else:
                stepSize = eps
        
            self.Q[armIndex]+= stepSize*(reward - self.Q[armIndex])
        
            if (not isStationary):
                self.walkAllArms()

            averageReward = cumulativeReward/numberOfPulls
            averageRewardArray.append(averageReward)
            rewardArray.append(reward)
            
            if (armIndex == self.bandit.bestArm()):
                optimalActionArray.append(1)
            else:
                optimalActionArray.append(0)

            averageOptimalAction = cumulativeOptimalAction/numberOfPulls
            optimalActionPctArray.append(averageOptimalAction)
            
        #averageReward = cumulativeReward/numberOfPulls
        print("==== Average Reward: " + str(averageReward))

        #averageOptimalAction = cumulativeOptimalAction/numberOfPulls
        print("==== Average Optimal Action: " + str(averageOptimalAction))

        return rewardArray, optimalActionArray
        
    #Change the mean value for each arm by a small amount. Defaulting to 0 with a variance of 0.01
    def walkAllArms(self, meanWalkLength=0, walkVariance=0.01):
        for arm in self.bandit.arms:
            arm.walk(meanWalkLength, walkVariance)
        
    def reset(self):
        self.Q = [0.0]*self.numberOfArms
        self.bandit = Bandit(self.numberOfArms, 0, 1)
    
    def MSE(self):
        error = 0
        for arm in range(self.numberOfArms):
            error+=np.square(self.bandit.arms[arm].meanReturn - self.Q[arm]) / self.numberOfArms
        return error
    
    def printErrorOfEstimate(self):
        for arm in range(self.numberOfArms):
            print("Arm actual value: " + str(self.bandit.arms[arm].meanReturn) + ", estimate: " + str(self.Q[arm]))
        print("MSE: " + str(self.MSE()))
        
    def testBestArmThroughWalks(self, numberOfPulls):
        print("Beginning tests. Initializing arms")
        self.resetQ()
        i=0
        for pull in range(numberOfPulls):
            if (i%1000 ==0):
                printBestArm()
            self.walkAllArms()
            i+=1

    def printBestArm(self):
        bestArm = self.bandit.bestArm()
    
        print("Best arm : " + str(bestArm))
    
    