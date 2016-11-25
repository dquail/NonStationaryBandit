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
algorithm.epsilonGreedyLearnMultipleRuns(2000, 1000, False, 0.1, 0.2)

"""


class BanditLearn:
    def __init__(self):
        self.numberOfArms = 10

        self.Q = [0.0]*self.numberOfArms
        self.bandit = Bandit(self.numberOfArms, 0, 1)

    def epsilonGreedyLearnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary = True, eps=0.5, alpha = -1):
        avgRewardVector = np.array([0.0]*numberOfPulls)
        optimalActionVector = np.array([0.0]*numberOfPulls)
        for run in range(numberOfRuns):
            if run % 100 == 0:
                print("Executing run " + str(run))
            self.reset()
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
        ax.set_ylabel('Optimal Action')
        ax.plot(optimalActionVector)
        plt.show()

        return (avgRewardVector, optimalActionVector)

    #def UCBLearnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary=True, eps=0.5, alpha=-1, c=-1):
    def gradientLearnMultipleRuns(self, numberOfRuns, numberOfPulls, alpha, isStationary = True):
        avgRewardVector = np.array([0.0] * numberOfPulls)
        optimalActionVector = np.array([0.0] * numberOfPulls)
        for run in range(numberOfRuns):
            if run % 100 == 0:
                print("Executing run " + str(run))
            self.reset()
            #def learn(self, numberOfPulls, isStationary = True, eps=0.5, alpha = -1, c = -1, isGradient=False):
            #learnResults = self.learn(numberOfPulls, isStationary, eps, alpha, c)
            learnResults = self.learn(numberOfPulls, isStationary, 1.0, alpha, -1, True)
            avgRewardVector += np.array(learnResults[0])
            optimalActionVector += np.array(learnResults[1])
        avgRewardVector = avgRewardVector / numberOfRuns
        optimalActionVector = optimalActionVector / numberOfRuns

        fig = plt.figure()
        fig.suptitle('Bandit', fontsize = 14, fontweight = 'bold')
        ax = fig.add_subplot(111)

        titleLabel = "Stationary: " + str(isStationary) + ", alpha:" + str(alpha)
        ax.set_title(titleLabel)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Optimal action %')
        ax.plot(optimalActionVector)
        plt.show()

        return (avgRewardVector, optimalActionVector)
        
    def UCBLearnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary=True, alpha=-1, c=-1):      
  
        avgRewardVector = np.array([0.0] * numberOfPulls)
        optimalActionVector = np.array([0.0] * numberOfPulls)
        for run in range(numberOfRuns):
            if run % 100 == 0:
                print("Executing run " + str(run))
            self.reset()
            #def learn(self, numberOfPulls, isStationary = True, eps=0.5, alpha = -1, c = -1):
            #learnResults = self.learn(numberOfPulls, isStationary, eps, alpha, c)
            learnResults = self.learn(numberOfPulls, isStationary, 1.0, alpha, c)
            avgRewardVector += np.array(learnResults[0])
            optimalActionVector += np.array(learnResults[1])
        avgRewardVector = avgRewardVector / numberOfRuns
        optimalActionVector = optimalActionVector / numberOfRuns

        fig = plt.figure()
        fig.suptitle('Bandit', fontsize = 14, fontweight = 'bold')
        ax = fig.add_subplot(111)

        titleLabel = "Stationary: " + str(isStationary) + ", alpha:" + str(alpha)
        ax.set_title(titleLabel)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Optimal action %')
        ax.plot(optimalActionVector)
        plt.show()

        return (avgRewardVector, optimalActionVector)

    def optimisticLearnMultipleRuns(self, numberOfRuns, numberOfPulls, initialEstimates, isStationary=True, alpha=-1):
        eps = 0.0 #never explores
        avgRewardVector = np.array([0.0]*numberOfPulls)
        optimalActionVector = np.array([0.0]*numberOfPulls)
        for run in range(numberOfRuns):
            if run % 100 == 0:
                print("Executing run " + str(run))
            self.reset()
            self.Q = [initialEstimates]*self.numberOfArms            
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
        ax.set_ylabel('Optimal Action')
        ax.plot(optimalActionVector)
        plt.show()

        return (avgRewardVector, optimalActionVector)

    """
    In all of the learn functions:
    Returns 2 arrays.
    - The first array is the rewards received at each time step
    - The second array contains the optimal action selection. 
    -- 0 if non optimal
    -- 1 if optimal action
    
    In this way, learn can be called multiple times on multiple bandit test beds to receive
    the average of returns at each time step as well as the average optimal 
    """
    def learn(self, numberOfPulls, isStationary = True, eps=0.5, alpha = -1, c = -1, isGradient = False):

        #For statistics and calculations
        rewardArray = []
        eachExploreArray = [0] * self.numberOfArms
        cumulativeReward = 0
        #averageRewardArray = []
        #cumulativeOptimalAction = 0
        #optimalActionPctArray = []
        numberOfPullsArray = [0]*self.numberOfArms
        optimalActionArray=[]        

        #For Gradient
        H = [0.0]*self.numberOfArms
        policyArray = [1/self.numberOfArms] * self.numberOfArms        
        
        #For UCB
        priorityArray = [0]*self.numberOfArms 

        for pull in range(numberOfPulls):
            #print("Q: " + str(self.Q))
            #Pick an action/arm to pull
            armIndex = 0            
            #Decide to explore vs. Exploit
            randomE = random()
            if (randomE < eps):
                #explore
                if c>0:
                    #Upper confidence bound action selection
                    #A=argmax(Q(a)+c*(sqrt(log(t)/N(a))
                    A = []
                    currentBestArm = argmax(self.Q)
                    for i in range(len(self.bandit.arms)):
                        if numberOfPullsArray[i] > 0:
                            val = self.Q[i] + c * np.sqrt((np.log(pull) / numberOfPullsArray[i]))
                        else:
                            #We want this to be a maximizing action
                            val = 10000 
                        #if (i == currentBestArm):
                        if(False):
                            #We never want to "explore" the best arm. So set it's A value to very low
                            A.append(-10000)
                        else:
                            A.append(val)
                    armIndex = argmax(A)
                    #print("Experimenting arm: " + str(armIndex))
                    #print("A: " + str(A))
                    #eachExploreArray[armIndex]+=1
                    #print("Arm explore array: ")
                    #print(eachExploreArray)
                    #print("Arm pull array: ")
                    #print(numberOfPullsArray)
                    #Comment below out after testing. With it in place, epsilonGreedy and UCB should be identical
                    #print("Chose: " + str(armIndex))
                    #print("Optimal: " + str(self.bandit.bestArm()))
                    #armIndex = randint(0,self.numberOfArms)
                elif (isGradient):
                    gradientPolicyDenomonator = 0
                    for j in range(self.numberOfArms):
                        gradientPolicyDenomonator+=np.exp(H[j])
                    for j in range(self.numberOfArms):
                        policyArray[j] = np.exp(H[j]) / gradientPolicyDenomonator
                    """
                    print("H array: ")
                    print (H)
                    print("Den: " + str(gradientPolicyDenomonator))
                    print("Policy array:")
                    print(policyArray)
                    print("Sum of policy array:")
                    print(str(np.sum(policyArray)))
                    print("======= Pull " + str(pull))
                    """
                    #armIndex = randint(0,numberOfArms)
                    armIndex = np.random.choice(np.arange(0, self.numberOfArms), p=policyArray)
                    H[armIndex] += 0.01
                    
                else:
                    #Eps greedy
                    #Simple epsilon greedy selection of exploring action
                    armIndex = randint(0,self.numberOfArms)
                

            else:
                #Exploit / Choose the best current action
                armIndex= argmax(self.Q)
                #print("Exploiting:" + str(armIndex))
            
            #Pull the lever
            reward = self.bandit.arms[armIndex].pull()
        
            #incriment the count of number of times it was used (in case we're using average)
            numberOfPullsArray[armIndex]+=1        
        
            #update statistics
            cumulativeReward+=reward
            
            #Update the Gradient priorities
            if (isGradient):
                averageReward = cumulativeReward / (pull+1)
                for i in range(self.numberOfArms):
                    if (not i == armIndex):
                        H[i]-= alpha *(reward - averageReward)*(policyArray[i])
                H[armIndex]+= alpha * (reward - averageReward)*(1 - policyArray[i])
                
            """
            if (armIndex == self.bandit.bestArm()):
                cumulativeOptimalAction+=1
            """
            #Update the Action values
            if (alpha==-1):
                stepSize = 1/numberOfPullsArray[armIndex]
            else:
                stepSize = alpha
        
            self.Q[armIndex]+= stepSize*(reward - self.Q[armIndex])
        
            if (not isStationary):
                self.walkAllArms()

            #averageReward = cumulativeReward/numberOfPulls
            #averageRewardArray.append(averageReward)
            rewardArray.append(reward)
            
            if (armIndex == self.bandit.bestArm()):
                optimalActionArray.append(1)
            else:
                optimalActionArray.append(0)

            #averageOptimalAction = cumulativeOptimalAction/numberOfPulls
            #optimalActionPctArray.append(averageOptimalAction)
            
        #averageReward = cumulativeReward/numberOfPulls
        #print("==== Average Reward: " + str(averageReward))

        #averageOptimalAction = cumulativeOptimalAction/numberOfPulls
        #print("==== Average Optimal Action: " + str(averageOptimalAction))
        #print(eachExploreArray)
        return rewardArray, optimalActionArray
        

    """
    Helper functions
    """

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
    
    