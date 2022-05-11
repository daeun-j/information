from Triplet import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# PATH = "2022_3/experiment/sort/" # for python console
'''
####################
#### Simulation  ###
####################
PATH = "sort/simulation_v3/" # for python console
datatype = 'Simulation'
submit_filename = "submission_trVAE.csv"
submit = pd.read_csv('{}{}_{}'.format(PATH, datatype, submit_filename))

labels = submit.Label*10+submit.Label_u
submit['new_label'] = submit.Label*10+submit.Label_u
# labels = np.sort(labels)
labels = np.unique(labels)
print(submit.head())
#for label in np.sort(labels)[::-1]:
for label in [13, 23, 33, 23, 12, 22,  11, 21, 31]:
    #if label in [11, 12, 13]: #11, 12, 13 21, 22, 23
        print(label)
        tmp = submit[submit["new_label"] == label].to_numpy()
        plt.scatter(tmp[:, 2], tmp[:, 3], marker='.', label=label)
        plt.xlim([0, 400])
        plt.ylim([-400, 400])

plt.legend()
plt.show()
'''

####################
####  Network    ###
####################
# PATH = "2022_3/simulation/data/network_v2/" # for python console
PATH = "sort/Network_v2/"
datatype = 'Network'
submit_filename = "submission_trVAE.csv"
print(PATH, datatype, submit_filename)
submit = pd.read_csv('{}{}_{}'.format(PATH, datatype, submit_filename))

labels = submit.Label*10+submit.Label_u
submit['new_label'] = submit.Label*10+submit.Label_u
# labels = np.sort(labels)
labels = np.unique(labels)
for label in np.sort(labels):#[::-1]:
    #print(label)
    #if label in [20, 21, 22]:
        tmp = submit[submit["new_label"] == label].to_numpy()
        plt.scatter(tmp[:, 4], tmp[:, 5], marker='.', label=label)
        plt.xlim([-20, 400])
        plt.ylim([-400, 400])
        plt.legend()
        plt.show()
