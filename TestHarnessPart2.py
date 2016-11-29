from BanditLearn import *


tB = BanditLearn()
# epsilonGreedyLearnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary=True, eps=0.5, alpha=-1):

returnResults = []
results = []

epsValues = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]#, 1/2, 1, 2, 4]
stationary = [False, False, False, False, False, False, False, False, False, False]#[True, True, True, True, True, True, True, True]#
numberOfPulls = 200000;
startOfAverage = 100000;
for i in range(0, len(epsValues)):
    tempResults = tB.epsilonGreedyLearnMultipleRuns(100, numberOfPulls, stationary[i], epsValues[i], 0.1)
    rewardSum = 0
    results = tempResults[0]
    returnResults.append(np.mean(results[startOfAverage-1:numberOfPulls]))
    #for j in range(startOfAverage, len(results)):
     #   rewardSum += results[j]
    #returnResults.append(rewardSum/startOfAverage)
print(returnResults)


fig = plt.figure(1)
fig.suptitle('Bandit', fontsize=14, fontweight='bold')
ax = fig.add_subplot(211)
#titleLabel = "Stationary: " + str(isStationary) + ", eps:" + str(eps) + ", alpha:" + str(alpha)
#titleLabel = "Average Return Over 2000 Bandits with 1000 pulls each"
#ax.set_title(titleLabel)
ax.set_xlabel('Epsilon')
ax.set_ylabel('Average Reward')
for i in range(0, len(epsValues)):
    if (epsValues[i] == -1 and stationary[i] == True):
        ax.plot(epsValues[i], returnResults[i], label="1/n (Stat.)")
    elif(epsValues[i] == -1):
        ax.plot(epsValues[i], returnResults[i], label="1/n")
    else:
        ax.plot(epsValues[i], returnResults[i], 'bo', label="Test")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
ax.set_xticks(epsValues[0:6])
plt.show()
