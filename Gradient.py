from Bandit import *
from pylab import *
import matplotlib.pyplot as plt

class Gradient:
    #eps of -1 will use an average reward (1/n) rathr than a constant step size
    def __init__(self, bandit, alpha):
        self.name = "Gradient"
        self.bandit = bandit
        self.numberOfArms = len(self.bandit.arms)
        self.numberOfPullsArray = [0]*self.numberOfArms
        self.cumulativeReward = 0
        
        self.Q = [0.0]*self.numberOfArms
        self.H = [0.0]*self.numberOfArms
        
        self.policyArray = [1/self.numberOfArms] * self.numberOfArms        
        self.alpha = alpha

        
    def learn(self, reward, armIndex):
        #Update the Action values
        self.numberOfPullsArray[armIndex]+=1
        self.cumulativeReward+=reward

        averageReward = self.cumulativeReward / np.sum(self.numberOfPullsArray)
        for i in range(self.numberOfArms):
            if (not i == armIndex):
                self.H[i]-= self.alpha *(reward - averageReward)*(self.policyArray[i])
            else: 
                self.H[i]+= self.alpha * (reward - averageReward)*(1 - self.policyArray[i])
        #print("H:" )
        #print(H)
        
        
    def policy(self):
        armIndex = 0            
        gradientPolicyDenomonator = 0
        for j in range(self.numberOfArms):
            gradientPolicyDenomonator+=np.exp(self.H[j])
            #print("Denom: " + str(gradientPolicyDenomonator))
        for j in range(self.numberOfArms):
            self.policyArray[j] = np.exp(self.H[j]) / gradientPolicyDenomonator

        armIndex = np.random.choice(np.arange(0, self.numberOfArms), p=self.policyArray)
        
        return armIndex
    
    def reset(self):
        self.numberOfPullsArray = [0]*self.numberOfArms
        self.Q = [0.0]*self.numberOfArms
        self.cumulativeReward = 0
        self.H = [0.0]*self.numberOfArms        
        self.policyArray = [1/self.numberOfArms] * self.numberOfArms
        
        
    
