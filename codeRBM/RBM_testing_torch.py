# %% code based on http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html

# imports and setup
import numpy as np
import torch
import matplotlib.pyplot as plt
from RBM_torch import RBM

plt.rcParams['figure.figsize'] = (15.0, 20.0)
plt.style.use('dark_background')

# whether to plot evolution of biases during training
PLOT_BIASES = False

# TESTING SSH GITHUB ACCESS

################################################################################

# load Ising model samples
dataOrig = np.load(
    "/home/fineline/projects/MEHTA_project/data"
    "/outputTest30000updates20000samples.npz")['arr_0']

numSamp, batchSize, = dataOrig.shape[0], 100
n_v, n_h = dataOrig.shape[1] * dataOrig.shape[2], 400

data = np.reshape(dataOrig, (numSamp, n_v)).T
# need original data for later, so make copy to feed to rbm (which modifies it)
data1st = np.copy(data)
print("data1st shape =", str(data1st.shape))

# setup torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%############################################################################
# FIRST LAYER RBM
################################################################################
print("Layer 1/3")
rbm1st = RBM(n_v, n_h, numSamp, batchSize)
numEpochs, learnRate, regWeight, mom, logInt = 50, 0.1, 0.008, 0.5, 1



## REALLY SLOW. Load data below instead
# train for ___ epochs, with learning rate 0.1
W_ijs1st, aa1st, bb1st = rbm1st.train(data1st, numEpochs, learnRate,
                                      biasesTo0=True,
                                      allParams=False,
                                      l1RegWeight=regWeight,
                                      momentum=mom,
                                      log_interval=logInt)
np.savez_compressed("data/couplingsL1.npz", W_ijs1st, aa1st, bb1st)

# W_ijs1st = np.load("data/couplingsL1.npz")['arr_0']
# aa1st = np.load("data/couplingsL1.npz")['arr_1']
# bb1st = np.load("data/couplingsL1.npz")['arr_2']
# rbm1st.setParams(W_ijs1st, aa1st, bb1st)

if PLOT_BIASES:
    # Vis unit biases (a)
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(aa1st.reshape(40, 40))
    plt.show()

    # Hid unit biases (b)
    plt.rcParams['figure.figsize'] = (15.0, 10.0)
    plt.imshow(bb1st.reshape(20, 20))
    plt.show()

# Plot wijs
wijs1st = W_ijs1st.T
plt.rcParams['figure.figsize'] = (20.0, 20.0)
print("wijs1st shape =", wijs1st.shape)
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(wijs1st[i].reshape(40, 40))
plt.show()

# %%############################################################################
# SECOND LAYER RBM
################################################################################
print("Layer 2/3")
# data for input to 2nd layer RBM
data2nd = rbm1st.vToh(data1st)
# setup 2nd RBM
numSamp, batchSize, n_v2, n_h2 = dataOrig.shape[0], 100, 400, 100
rbm2nd = RBM(n_v2, n_h2, numSamp, batchSize)
numEpochs, learnRate, regWeight, mom, logInt = 100, 0.1, 0.008, 0.5, 1

## PRETTY SLOW. Load data below instead
# train for ___ epochs, with learning rate 0.1
W_ijs2nd, aa2nd, bb2nd = rbm2nd.train(data2nd, numEpochs, learnRate, True,
                                      False, regWeight, mom, logInt)
np.savez_compressed("data/couplingsL2.npz", W_ijs2nd, aa2nd, bb2nd)

W_ijs2nd = np.load("data/couplingsL2.npz")['arr_0']
aa2nd = np.load("data/couplingsL2.npz")['arr_1']
bb2nd = np.load("data/couplingsL2.npz")['arr_2']
rbm2nd.setParams(W_ijs2nd, aa2nd, bb2nd)

if PLOT_BIASES:
    # Vis unit biases (a)
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(aa2nd.reshape(20, 20))
    plt.show()

    # Hid unit biases (b)
    plt.rcParams['figure.figsize'] = (10.0, 5.0)
    plt.imshow(bb2nd.reshape(10, 10))
    plt.show()

# %%
wijs2nd = W_ijs2nd.T
plt.rcParams['figure.figsize'] = (15.0, 20.0)
print("wijs2nd shape =", wijs2nd.shape)
for i in range(20):
    plt.subplot(10, 10, i + 1)
    plt.imshow(wijs2nd[i].reshape(20, 20))
plt.show()

# %%
# reconstruction of full receptor fields
recept2nd = [(1 / 4e2) * np.dot(wijs2nd[i, :], wijs1st) for i in range(n_h2)]

plt.rcParams['figure.figsize'] = (15.0, 20.0)
for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.imshow(recept2nd[i].reshape(40, 40))
plt.show()

# %%############################################################################
# THIRD LAYER RBM
################################################################################
print("Layer 3/3")
# data for input to 3rd layer RBM
data3rd = rbm2nd.vToh(data2nd)
# setup 3nd RBM
numSamp, batchSize, n_v3, n_h3 = dataOrig.shape[0], 100, 100, 25
rbm3rd = RBM(n_v3, n_h3, numSamp, batchSize)
# TODO: do an SMO over these parameters
# numEpochs, learnRate, regWeight, mom, logInt = 100, 0.1, 0.008, 0.6, 1
numEpochs, learnRate, regWeight, mom, logInt = 100, 0.2, 0.0008, 0.9, 1

## SLOWISH. Load data below instead
# train for numEpochs, at learnRate
W_ijs3rd, aa3rd, bb3rd = rbm3rd.train(data3rd, numEpochs, learnRate, True,
                                      False, regWeight, mom, logInt)
np.savez_compressed("data/couplingsL3.npz", W_ijs3rd, aa3rd, bb3rd)

# W_ijs3rd = np.load("data/couplingsL3.npz")['arr_0']
# aa3rd    = np.load("data/couplingsL3.npz")['arr_1']
# bb3rd    = np.load("data/couplingsL3.npz")['arr_2']
# rbm3rd.setParams(W_ijs3rd, aa3rd, bb3rd)

if PLOT_BIASES:
    # Vis unit biases (a)
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(aa3rd.reshape(10, 10))
    plt.show()

    # Hid unit biases (b)
    plt.rcParams['figure.figsize'] = (10.0, 5.0)
    plt.imshow(bb3rd.reshape(5, 5))
    plt.show()

wijs3rd = W_ijs3rd.T
plt.rcParams['figure.figsize'] = (15.0, 20.0)
print("wijs3rd shape=", wijs3rd.shape)
for i in range(25):
    plt.subplot(10, 10, i + 1)
    plt.imshow(wijs3rd[i].reshape(10, 10))
plt.show()

# reconstruction of full receptor fields
recept3rd = [(1 / 4e4) * np.dot(np.dot(wijs3rd[i, :], wijs2nd), wijs1st) for i
             in range(n_h3)]

plt.rcParams['figure.figsize'] = (15.0, 8.0)
for i in range(25):
    plt.subplot(3, 10, i + 1)
    plt.imshow(recept3rd[i].reshape(40, 40))
plt.show()

# %%############################################################################
# RECONSTRUCT DATA (improperly)
################################################################################

# NB we should only be using the final layer weights (hidL3), and repropagating
# them back through the layers, rather than taking a sum over all layers,
# as we do here.
recon_num = 5
dataL = np.copy(data[:, :recon_num])

hidL1 = rbm1st.vToh(dataL)
hidL2 = rbm2nd.vToh(hidL1)
hidL3 = rbm3rd.vToh(hidL2)

# reconstruction NB: previously I was just using dataReconONE below,
# which only gave the low freq part of the reconstruction. adding
# dataReconTWO and dataReconTHREE gives the higher freq components
dataReconONE = np.dot(wijs1st.T,
                      np.dot(wijs2nd.T, wijs3rd.T.dot(hidL3))).reshape(
    (40, 40, 5))
dataReconTWO = np.dot(wijs1st.T, wijs2nd.T.dot(hidL2)).reshape((40, 40, 5))
dataReconTHREE = wijs1st.T.dot(hidL1).reshape((40, 40, 5))

# Not sure about the numerical weighting factors here...
dataRecon = dataReconONE + 100 * dataReconTWO + 400 * 100 * dataReconTHREE

for i in range(recon_num):
    # reconstructed samples
    plt.subplot(3, recon_num, i + 1)
    plt.imshow(dataRecon[:, :, i])
    # reconstructed samples binarized
    plt.subplot(3, recon_num, i + 1 + recon_num)
    plt.imshow(dataRecon[:, :, i] > 0)
    # original samples
    plt.subplot(3, recon_num, i + 1 + 2 * recon_num)
    plt.imshow(dataL[:, i].reshape(40, 40))
plt.show()

# %%############################################################################
# RECONSTRUCT ALL DATA POINTS
################################################################################

recon_num = 5
dataL = np.copy(data[:, :recon_num])

hidL1 = rbm1st.vToh(dataL)
hidL2 = rbm2nd.vToh(hidL1)
hidL3 = rbm3rd.vToh(hidL2)
hidL2_back = rbm3rd.hTov(hidL3)
hidL1_back = rbm2nd.hTov(hidL2_back)
dataL_back = rbm1st.hTov(hidL1_back)

for i in range(recon_num):
    # reconstructed samples
    dataL_backRes = dataL_back[:, i].reshape(40, 40)
    plt.subplot(3, recon_num, i + 1)
    plt.imshow(dataL_backRes)
    # reconstructed samples binarized
    plt.subplot(3, recon_num, i + 1 + recon_num)
    plt.imshow(dataL_backRes > 0.5)
    # original samples
    plt.subplot(3, recon_num, i + 1 + 2 * recon_num)
    plt.imshow(dataL[:, i].reshape(40, 40))
plt.show()
