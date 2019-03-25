# RBM implementation in pytorch
# TODO: put in '.to(device)' on all relevant variables

import numpy as np
from tqdm import tqdm # progress monitor
import torch


class RBM:
  def __init__(self, numVis, numHid, numTrain, bs):
    self.n_v = numVis
    self.n_h = numHid
    self.m = numTrain
    self.bs = bs
    self.trained = False

  def setParams(self, W, a, b):
    ''' Sets RBM parameters.
          Inputs: "W" - (n_v x n_h) matrix of trained couplings
                  "a" - (n_v x 1) vector of trained visible biases
                  "b" - (n_h x 1) vector of trained hidden biases
    '''
    self.w_ij, self.a, self.b = W, a, b
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
    # number of samples must be integer multiple of bs
    assert (np.mod(data.shape[1], self.bs) == 0)

    batchesInSample = self.m / self.bs

    # Init biases
    if biasesTo0 is True:
      self.a, self.b = torch.zeros(self.n_v, 1), torch.zeros(self.n_h, 1)
    else:
      # fraction of samples with i'th spin on (HINTON section 8)
      vp = 1. * torch.sum(data, dim=1, keepdim=True) / self.m
      self.a, self.b = torch.log(vp / (1. - vp)), torch.ones(self.n_h, 1)

    # initialize weights to gaussian small values (HINTON)
    # TODO
    np_rng = np.random.RandomState(1234)
    self.w_ij = np_rng.normal(0, 0.01, size=(self.n_v, self.n_h))

    # Placeholder for momentum
    v = torch.zeros(self.w_ij.shape)

    # For all w_ijs, as, and bs in run
    w_ijs, aa, bb = [], [], []

    for i in tqdm(range(numEpochs)):
      # randomize sample order for this epoch
      dataThisEpoch = torch.clone(data)
      dataThisEpoch = dataThisEpoch[:, torch.randperm(self.m)]

      # make batches for this epoch
      batches = torch.split(dataThisEpoch, batchesInSample, dim=1)

      for batch in batches:
        # probability that hidden unit is 1
        # Gives (n_h x bs) matrix
        pHidData = self._logistic(torch.dot(self.w_ij.transpose(0, 1), batch) +
                                  self.b)

      # draw a sample from pHidData
      ber = torch.distributions.Bernoulli()

      randProb = torch.rand()
      sampHidData = np.random.binomial(1, pHidData)

      # reconstructed visible pdf from the hidden data sample
      pVisRecon = self._logistic(torch.dot(self.w_ij, sampHidData) + self.a)


  def _logistic(self, x):
    return 1.0 / (1 + torch.exp(-x))











