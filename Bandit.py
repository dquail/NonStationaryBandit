from pylab import zeros, normal, random
from Arm import *

class Bandit:

    def __init__(self, numberOfArms, armMean, variance):
        self.arms = []
        for arm in range(numberOfArms):
            armRealMean = normal(armMean, variance)
            arm = Arm(armRealMean, variance)
            self.arms.append(arm)
                
    def walk(self, meanStep, varianceStep):
        for arm in self.arms:
            arm.walk(meanStep, varianceStep)
        
    def printBandit(self):
        i = 0
        for arm in self.arms:
            print("Arm " + str(i) + ":")
            arm.printCurrentValues()
            i+=1