# Maya Z Wang 07 2020

# this program load the PCA permutation and bootstrap results

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from PCA_functions import *
funcs = PCA_functions
##  ############## figure out task parameters #################
##
# use sessions: 13, 39 (actual session # not index)
dat = pickle.load(open("ses13.p", "rb"))
# dat = pickle.load(open("ses39.p", "rb"))
spk = dat["spks"]
# print(dat.keys())

# left recorded
rp = np.array((dat['response']))

choose_l = np.array((dat['response'] == 1))
choose_r = np.array((dat['response'] == -1))

nogo = np.array((dat['response'] == 0))
go = np.array((dat['response'] != 0))

contrast_l = dat['contrast_left']
contrast_r = dat['contrast_right']

contrast_l_high = contrast_r < contrast_l
contrast_r_high = contrast_r > contrast_l
contrast_equal = contrast_r == contrast_l

correct = np.zeros_like(choose_l)
for i in range(len(choose_l)):
    if contrast_l[i] > contrast_r[i] and rp[i] == 1:
        correct[i] = 1
    elif contrast_l[i] < contrast_r[i] and rp[i] == -1:
        correct[i] = 1
    elif contrast_l[i] == contrast_r[i] and rp[i] == 0:
        correct[i] = 1

##
# binSize = 100
binSize = 50
print('go-cue time: ' + str(np.median(dat['gocue']*1000/100)+500/binSize))
print('response time: ' + str(np.median(dat['response_time']*1000/binSize)+500/binSize))
print('feedback time: ' + str(np.median(dat['feedback_time']*1000/binSize)+500/binSize))
##
allAreas = dat["brain_area"]
usedAreas = ['CA1', 'VISam', 'PL', 'MOs']
area = dict()
cellN = np.zeros(4)
for i in range(len(usedAreas)):
    areaId = allAreas == usedAreas[i]
    tmp_Area = spk[areaId, :, :]
    print(tmp_Area.shape)
    area[i] = funcs.frNormalization(tmp_Area)
    cellN[i] = tmp_Area.shape[0]

##
iA = 3
stepSize=10
# dd = funcs.moveMean_step(stepSize, area[iA])
iN = area[iA].shape[0]
iT = area[iA].shape[1]
new_n_bins = int(area[0].shape[2]/stepSize)
dd=funcs.bin_ndarray(area[iA],new_shape=(iN,iT,new_n_bins),operation='mean') / 0.1
print(dd.shape)
##
for i in range(dd.shape[1]):
    if i == 0:
        fr_mat = np.rollaxis(dd[:, i, :], 1, 0)
    else:
        tmp = np.rollaxis(dd[:, i, :], 1, 0)
        fr_mat = np.append(fr_mat, tmp, axis=0)
print(np.unique(fr_mat))
## separate trials by contras
# np.intersect1d(ar1, ar2, assume_unique=False)
trlLen = 25
trlStarts = np.arange(1, len(fr_mat), trlLen)

compare = 3

if compare == 1:
    contrast_rl = contrast_l < 0.45
    contrast_rh = contrast_l > 0.45
elif compare == 2:
    contrast_rl = choose_l
    contrast_rh = choose_r
elif compare == 3:
    contrast_rl = go
    contrast_rh = nogo

trl_rl = trlStarts[contrast_rl]-1
trl_rh = trlStarts[contrast_rh]-1
print(len(trl_rl))
print(len(trl_rh))
##
tmpmat = np.zeros([trlLen, fr_mat.shape[1], len(trl_rl)])
for i in range(len(trl_rl)):
    tmpmat[:, :, i]=fr_mat[trl_rl[i]:trl_rl[i]+trlLen, :]
rlmat = np.mean(tmpmat, axis=2)


tmpmat2 = np.zeros([trlLen, fr_mat.shape[1], len(trl_rh)])
for i in range(len(trl_rh)):
    tmpmat2[:, :, i]=fr_mat[trl_rh[i]:trl_rh[i]+trlLen, :]
rhmat = np.mean(tmpmat2, axis=2)


pcamat=np.vstack((rlmat, rhmat))
# print(np.unique(rlmat))
# print(np.unique(rhmat))
print(pcamat.shape)

##
score, _, _ = funcs.pca(pcamat)
x1=score[0:trlLen, 0]
y1=score[0:trlLen, 1]
z1=score[0:trlLen, 2]

x2=score[trlLen:, 0]
y2=score[trlLen:, 1]
z2=score[trlLen:, 2]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x1, y1, z1, 'Red')
ax.plot3D(x2, y2, z2, 'Blue')

ax.text(x1[0], y1[0], z1[0], 'start')
ax.text(x1[-1], y1[-1], z1[-1], 'end')
# ax.scatter3D(x1, y1, z1, c=x1, cmap='Reds')
# ax.scatter3D(x1, y1, z1, c=y1, cmap='Reds')
# ax.scatter3D(x1, y1, z1, c=z1, cmap='Reds')

ax.text(x2[0], y2[0], z2[0], 'start')
ax.text(x2[-1], y2[-1], z2[-1], 'end')
# ax.scatter3D(x2, y2, z2, c=x2, cmap='Blues')
# ax.scatter3D(x2, y2, z2, c=y2, cmap='Blues')
# ax.scatter3D(x2, y2, z2, c=z2, cmap='Blues')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

if compare == 1:
    plt.legend(('contrast low', 'contrast high'), loc='lower left')
    plt.title('Population trajectories in ' + usedAreas[iA] + ' on left contrast', y=1.1)
elif compare == 2:
    plt.legend(('choose left', 'choose right'), loc='lower left')
    plt.title('Population trajectories in ' + usedAreas[iA] + ' on L/R Choice', y=1.1)
elif compare == 3:
    plt.legend(('go', 'no-go'), loc='lower left')
    plt.title('Population trajectories in ' + usedAreas[iA] + ' on Go/NoGo', y=1.1)



##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs[0].add_collection(lc)
fig.colorbar(line, ax=axs[0])

