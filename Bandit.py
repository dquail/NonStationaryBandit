from pylab import *
from Arm import *

class Bandit:
    
    def __init__(self, numberOfArms, armMean, variance):
        self.numberOfArms = numberOfArms
        self.armMean = armMean
        self.variance = variance
        
        self.arms = []
        for arm in range(numberOfArms):
            armRealMean = normal(armMean, variance)
            arm = Arm(armRealMean, variance)
            self.arms.append(arm)
        self.bestArm = self.calculateBestArm()
                
    def reset(self):
        self.arms = []
        for arm in range(self.numberOfArms):
            armRealMean = normal(self.armMean, self.variance)
            arm = Arm(armRealMean, self.variance)
            self.arms.append(arm)
        self.bestArm = self.calculateBestArm()
        
    def walk(self, meanStep = 0, varianceStep = 0.01):
        for arm in self.arms:
            arm.walk(meanStep, varianceStep)
        self.bestArm = self.calculateBestArm()
        
    def printBandit(self):
        i = 0
        for arm in self.arms:
            print("Arm " + str(i) + ":")
            arm.printCurrentValues()
            i+=1
            
    def pull(self, armIndex):
        return self.arms[armIndex].pull()
        
    
    def calculateBestArm(self):
        armMeanValues = []
        for arm in self.arms:
            armMeanValues.append(arm.meanReturn)
        return argmax(armMeanValues)
        