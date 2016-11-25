from pylab import *

numberOfArms = 10
numberOfPulls = 100

H = [0.0]*numberOfArms
policyArray = [1/numberOfArms] * numberOfArms

for i in range(numberOfPulls):

    gradientPolicyDenomonator = 0
    for j in range(numberOfArms):
        gradientPolicyDenomonator+=np.exp(H[j])
    for j in range(numberOfArms):
        policyArray[j] = np.exp(H[j]) / gradientPolicyDenomonator
    print("H array: ")
    print (H)
    print("Den: " + str(gradientPolicyDenomonator))
    print("Policy array:")
    print(policyArray)
    print("Sum of policy array:")
    print(str(np.sum(policyArray)))
    print("======= Pull " + str(i))
    #armIndex = randint(0,numberOfArms)
    armIndex = np.random.choice(np.arange(0, numberOfArms), p=policyArray)
    H[armIndex] += 0.01
    
    
    