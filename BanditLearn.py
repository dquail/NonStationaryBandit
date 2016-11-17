from Bandit import *
from pylab import *
numberOfArms = 10

Q = [0.0]*numberOfArms
bandit = Bandit(numberOfArms, 0, 1)

def play(numberOfPulls, isStationary = True, eps=0.5, alpha = -1):
    
    reset()

    cumulativeReward = 0
    cumulativeOptimalAction = 0
    
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
        
        #incriment the count of number of times it was used (in case we're using average)
        numberOfPullsArray[armIndex]+=1        
        
        #update statistics
        cumulativeReward+=reward

        if (armIndex == bandit.bestArm()):
            cumulativeOptimalAction+=1
        
        #Update the Action values
        if (alpha==-1):
            stepSize = 1/numberOfPullsArray[armIndex]
        else:
            stepSize = eps
        
        Q[armIndex]+= stepSize*(reward - Q[armIndex])
        
        if (not isStationary):
            walkAllArms()

    averageReward = cumulativeReward/numberOfPulls
    print("==== Average Reward: " + str(averageReward))

    averageOptimalAction = cumulativeOptimalAction/numberOfPulls
    print("==== Average Optimal Action: " + str(averageOptimalAction))

    return averageReward, averageOptimalAction
        
#Change the mean value for each arm by a small amount. Defaulting to 0 with a variance of 0.01
def walkAllArms(meanWalkLength=0, walkVariance=0.01):
    for arm in bandit.arms:
        arm.walk(meanWalkLength, walkVariance)
        
def reset():
    Q = [0.0]*numberOfArms
    bandit = Bandit(numberOfArms, 0, 1)
    
def MSE():
    error = 0
    for arm in range(numberOfArms):
        error+=np.square(bandit.arms[arm].meanReturn - Q[arm]) / numberOfArms
    return error
    
def printErrorOfEstimate():
    for arm in range(numberOfArms):
        print("Arm actual value: " + str(bandit.arms[arm].meanReturn) + ", estimate: " + str(Q[arm]))
    print("MSE: " + str(MSE()))
        
def testBestArmThroughWalks(numberOfPulls):
    print("Beginning tests. Initializing arms")
    resetQ()
    i=0
    for pull in range(numberOfPulls):
        if (i%1000 ==0):
            printBestArm()
        walkAllArms()
        i+=1

def printBestArm():
    bestArm = bandit.bestArm()
    
    print("Best arm : " + str(bestArm))
    
    