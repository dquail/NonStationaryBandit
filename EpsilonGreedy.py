from Bandit import *
from pylab import *
import matplotlib.pyplot as plt

class EpsilonGreedy:
    #eps of -1 will use an average reward (1/n) rathr than a constant step size
    def __init__(self, bandit, alpha, eps):
        self.bandit = bandit
        self.numberOfArms = len(self.bandit.arms)
        self.numberOfPullsArray = [0]*self.numberOfArms

        self.Q = [0.0]*self.numberOfArms

        self.alpha = alpha
        self.eps = eps
        
    def learn(self, reward, armIndex):
        #Update the Action values
        self.numberOfPullsArray[armIndex]+=1

        if (self.alpha==-1):
            stepSize = 1/self.numberOfPullsArray[armIndex]
        else:
            stepSize = self.alpha

        #do the learning        
        self.Q[armIndex]+= stepSize*(reward - self.Q[armIndex])
      
        
    def policy(self):
        armIndex = 0            
        #Decide to explore vs. Exploit
        randomE = random()
        if (randomE < self.eps):
            armIndex = randint(0,self.numberOfArms)
        else:
            #Exploit / Choose the best current action
            armIndex= argmax(self.Q)

        return armIndex
    
    def reset(self):
        self.Q = [0.0]*self.numberOfArms
        self.numberOfPullsArray = [0]*self.numberOfArms
        
    
