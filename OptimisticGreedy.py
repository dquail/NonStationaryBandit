from Bandit import *
from pylab import *
import matplotlib.pyplot as plt

class OptimisticGreedy:
    #eps of -1 will use an average reward (1/n) rathr than a constant step size
    def __init__(self, bandit, initialValues, alpha):
        self.bandit = bandit
        self.numberOfArms = len(self.bandit.arms)
        self.numberOfPullsArray = [0]*self.numberOfArms

        self.Q = [initialValues]*self.numberOfArms

        self.alpha = alpha
        self.initialValues = initialValues
        
    def learn(self, reward, armIndex):
        #Update the Action values
        self.numberOfPullsArray[armIndex]+=1

        if (self.alpha==-1):
            stepSize = 1/numberOfPullsArray[armIndex]
        else:
            stepSize = self.alpha

        #do the learning        
        #print("Learning with reward: " + str(reward) + " armIndex: " + str(armIndex))
        #print("Q before learning: ")
        #print(self.Q)
        self.Q[armIndex]+= stepSize*(reward - self.Q[armIndex])
        #print("Q after learning: ")
        #print(self.Q)        

        
        
    def policy(self):
        armIndex = 0            
        #Decide to explore vs. Exploit
        armIndex= argmax(self.Q)
        return armIndex
    
    def reset(self):
        self.Q = [self.initialValues]*self.numberOfArms
        self.numberOfPullsArray = [0]*self.numberOfArms
        
        
    
