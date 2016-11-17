from pylab import zeros, normal, random

class Arm:
    
    def __init__(self, meanReturn, varianceReturn):
        self.meanReturn = meanReturn
        self.varianceReturn = varianceReturn
        
    def printCurrentValues(self):
        print ("Mean: " + str(self.meanReturn) + ", variance: " + str(self.varianceReturn))
        
    def walk(self, meanStep, varianceStep):
        walkSize = normal(meanStep,varianceStep)
        self.meanReturn+=walkSize
        return walkSize
        
    def pull(self):
        rValue = self.meanReturn
        
        if (self.varianceReturn>0):
            rValue = normal(self.meanReturn, self.varianceReturn)
        
        return rValue