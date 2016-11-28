from Bandit import *
from pylab import *
import matplotlib.pyplot as plt

class UCB:
    #eps of -1 will use an average reward (1/n) rathr than a constant step size
    def __init__(self, bandit, c, alpha):
        self.bandit = bandit
        self.numberOfArms = len(self.bandit.arms)
        self.numberOfPullsArray = [0]*self.numberOfArms

        self.Q = [0.0]*self.numberOfArms
        self.c = c
        self.alpha = alpha

        
    def learn(self, reward, armIndex):
        #Update the Action values
        self.numberOfPullsArray[armIndex]+=1

        if (self.alpha==-1):
            stepSize = 1/self.numberOfPullsArray[armIndex]
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
        #Upper confidence bound action selection
        #A=argmax(Q(a)+c*(sqrt(log(t)/N(a))
        A = []
        currentBestArm = argmax(self.Q)
        for i in range(len(self.bandit.arms)):
            if self.numberOfPullsArray[i] > 0:
                pull = np.sum(self.numberOfPullsArray)
                val = self.Q[i] + self.c * np.sqrt((np.log(pull) / self.numberOfPullsArray[i]))
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
        return armIndex
    
    def reset(self):
        self.Q = [0.0]*self.numberOfArms
        self.numberOfPullsArray = [0]*self.numberOfArms
        
        
    
