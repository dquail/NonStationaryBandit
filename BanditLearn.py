from Bandit import *
from pylab import *

class BanditLearn:
    def __init__(self):
        self.numberOfArms = 10

        self.Q = [0.0]*self.numberOfArms
        self.bandit = Bandit(self.numberOfArms, 0, 1)
        
    def play(self, numberOfPulls, isStationary = True, eps=0.5, alpha = -1):
    
        self.reset()

        cumulativeReward = 0
        cumulativeOptimalAction = 0
    
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
        print("==== Average Reward: " + str(averageReward))

        averageOptimalAction = cumulativeOptimalAction/numberOfPulls
        print("==== Average Optimal Action: " + str(averageOptimalAction))

        return averageReward, averageOptimalAction
        
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
    
    