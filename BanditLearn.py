from Bandit import *
from pylab import *
numberOfArms = 10

Q = [0.0]*numberOfArms
bandit = Bandit(numberOfArms, 0, 1)

def playStationary(numberOfPulls, eps):
    resetQ()
    cumulativeReward = 0
    numberOfPullsArray = [0]*numberOfArms
    for pull in range(numberOfPulls):

        #Pick an action/arm to pull
        armIndx = 0
        #Decide to explore vs. Exploit
        randomE = random()                
        if (randomE < eps):
            #explore
            armIndex = randint(0,numberOfArms)    
        else:
            #Exploit / Choose the best current action
            armIndex= argmax(Q)
            
        #Pull the lever
        reward = bandit.arms[armIndex].pull()
        cumulativeReward+=reward
        
        #incriment the count of number of times it was used (in case we're using average)
        numberOfPullsArray[armIndex]+=1
        
        Q[armIndex]+= 1/numberOfPullsArray[armIndex]*(reward - Q[armIndex])
    return cumulativeReward
        
def resetQ():
    Q = [0.0]*numberOfArms
    
def MSE():
    error = 0
    for arm in range(numberOfArms):
        error+=np.square(bandit.arms[arm].meanReturn - Q[arm])
    return error
    
def printErrorOfEstimate():
    for arm in range(numberOfArms):
        print("Arm actual value: " + str(bandit.arms[arm].meanReturn) + ", estimate: " + str(Q[arm]))
    print("MSE: " + str(MSE()))
        
#def playNonStationary(numberOfPulls, epsilon):
    