# RBM implementation in pytorch

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

class RBM:
  def __init__(self, numVis, numHid, numTrain, bs):
    self.n_v = numVis
    self.n_h = numHid
    self.m = numTrain
    self.bs = bs
    self.trained = False
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def setParams(self, W, a, b):
    ''' Sets RBM parameters.
          Inputs: "W" - (n_v x n_h) matrix of trained couplings
                  "a" - (n_v x 1) vector of trained visible biases
                  "b" - (n_h x 1) vector of trained hidden biases
    '''
    self.w_ij = torch.from_numpy(W).to(self.device)
    self.a = torch.from_numpy(a).to(self.device)
    self.b = torch.from_numpy(b).to(self.device)
    self.trained = True

  def train(self, data, numEpochs, trainRate, biasesTo0=False, allParams=False,
            l1RegWeight=0, momentum=0, log_interval=10):
    ''' Trains the RBM.
          Inputs:  "data"         - (n_v x m) array of inputs
                   "numEpochs"    - number of epochs to train for
                   "trainRate"    - training rate
                   "biasesTo0"    - if true, biases are left out of calculation
                   "allParams"    - if true, log a, b, and wij every
                                    "log_interval"
                   "l1RegWeight"  - strength of L1 regularization (no reg if 0)
                   "momentum"     - strength of momentum (no mom if 0)
                   "log_interval" - log progress every this number of epochs
          Returns: "w_ij" - (n_v x n_h) matrix of trained couplings
                   "a"    - (n_v x 1) vector of trained visible biases
                   "b"    - (n_h x 1) vector of trained hidden biases
    '''
    # make sure data is of specified shape
    assert (data.shape[0] == self.n_v)
    data_t = torch.from_numpy(data).to(self.device)
    # number of samples must be integer multiple of bs
    assert (np.mod(data.shape[1], self.bs) == 0)

    batchesInSample = self.m // self.bs

    # Init biases
    if biasesTo0 is True:
      self.a = torch.zeros(self.n_v, 1).type(torch.DoubleTensor).to(self.device)
      self.b = torch.zeros(self.n_h, 1).type(torch.DoubleTensor).to(self.device)
    else:
      # fraction of samples with i'th spin on (HINTON section 8)
      vp = 1. * torch.sum(data_t, dim=1, keepdim=True) / self.m
      self.a = torch.log(vp / (1. - vp))
      self.b = torch.ones(self.n_h, 1).type(torch.DoubleTensor).to(self.device)

    # initialize weights to gaussian small values (HINTON)
    np_rng = np.random.RandomState(1234)
    self.w_ij = np_rng.normal(0, 0.01, size=(self.n_v, self.n_h))
    self.w_ij = torch.from_numpy(self.w_ij).to(self.device)

    # Placeholder for momentum
    v = torch.zeros(self.w_ij.shape).type(torch.DoubleTensor).to(self.device)

    # For all w_ijs, as, and bs in run
    w_ijs, aa, bb = [], [], []

    for i in tqdm(range(numEpochs)):
      # randomize sample order for this epoch
      dataThisEpoch = torch.clone(data_t)
      dataThisEpoch = dataThisEpoch[:, torch.randperm(self.m)]

      # make batches for this epoch
      batches = torch.chunk(dataThisEpoch, batchesInSample, dim=1)

      for batch in batches:
        # probability that hidden unit is 1
        # Gives (n_h x bs) matrix
        pHidData = self._logistic(torch.mm(self.w_ij.t(), batch) + self.b)

        # random_probabilities = torch.rand_like(pHidData).to(self.device)
        # sampHidData = (pHidData >= random_probabilities).float()
        sampHidData = Bernoulli(pHidData).sample().to(self.device)

        # reconstructed visible pdf from the hidden data sample
        pVisRecon = self._logistic(torch.mm(self.w_ij, sampHidData) + self.a)

        # reconstructed hidden pdf
        pHidRecon = self._logistic(torch.mm(self.w_ij.t(), pVisRecon) + self.b)

        # <v h> correlations for data and reconstructed
        visHidCorrData = torch.mm(batch, pHidData.t()) / self.bs
        visHidCorrRecon = torch.mm(pVisRecon, pHidRecon.t()) / self.bs

        # gradient ascent on parameters, with opt L1 regularization
        # TODO check minus sign
        v = momentum * v + trainRate * (visHidCorrData - visHidCorrRecon -
                                        l1RegWeight * torch.sign(self.w_ij))

        self.w_ij += v
        if biasesTo0 is False:
          self.a += (trainRate / self.bs) * torch.sum(batch - pVisRecon,
                                                      dim=1, keepdim=True)
          self.b += (trainRate / self.bs) * torch.sum(pHidData - pHidRecon,
                                                      dim=1, keepdim=True)

      # log weights during training if 'allParams' is set
      if allParams == True and i % log_interval == 0:
        w_ijs.append(self.w_ij.cpu().numpy())
        aa.append(self.a.cpu().numpy())
        bb.append(self.b.cpu().numpy())

    # final result
    w_ijs.append(self.w_ij.cpu().numpy())
    aa.append(self.a.cpu().numpy())
    bb.append(self.b.cpu().numpy())

    self.trained = True
    return w_ijs, aa, bb


  def _logistic(self, x):
    return 1.0 / (1 + torch.exp(-x))

  def _logisticnp(self, x):
    return 1.0 / (1 + np.exp(-x))

  def vToh(self, vis):
    assert (self.trained == True)
    assert (vis.shape[0] == self.n_v)

    # Calculate final hidden activations from final model
    return self._logisticnp(np.dot(self.w_ij.cpu().numpy().T, vis) +
                            self.b.cpu().numpy())

  def hTov(self, hid):
    assert (self.trained == True)
    assert (hid.shape[0] == self.n_h)

    # Figure out what biases to put in here (compare vToH function, above)
    return self._logisticnp(np.dot(self.w_ij.cpu().numpy(), hid))











