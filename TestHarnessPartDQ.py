from BanditLearn import *


tb = BanditLearn()
# epsilonGreedyLearnMultipleRuns(self, numberOfRuns, numberOfPulls, isStationary=True, eps=0.5, alpha=-1):

returnResults = []
results = []

initialValues = [1/4, 1/2, 1, 2, 4]
stationary = True
#stationary = False
numberOfPulls = 1000
numberOfRuns = 2000
startOfAverage = 100000
for i in range(0, len(initialValues)):
    print("Testing value: " + str(initialValues[i]))
    tempResults = tb.optimisticLearnMultipleRuns(numberOfRuns, numberOfPulls, initialValues[i], stationary, eps=0, alpha=0.1) 
    rewardSum = 0
    results = tempResults[0]
    returnResults.append(np.mean(results))
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
ax.set_xlabel('Initial')
ax.set_ylabel('Average Reward')
for i in range(0, len(initialValues)):
    ax.plot(initialValues[i], returnResults[i], label="1/n (Stat.)")
"""
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
ax.set_xticks(epsValues[0:6])
"""
plt.show()
