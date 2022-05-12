# RBM implementation
#
# References (in all caps):
# HINTON - "2010 - Hinton - A Practical Guide to Training RBMs"

import numpy as np
from tqdm import tqdm

# RBM class
#
# - RBM has visible, binary, stochastic pixels which are connected to hidden,
#   binary, stochastic feature detectors using symmetrically weighted
#   connections.
# - Input vectors/training examples/visible units are column vectors of
#   dimension n_v
# - Hidden units are a column vector of dimension n_h.
# - A set of m training examples is represented as an (n_v x m) matrix.
#
# input to RBM constructor
#  - num vis units, n_v
#  - num hidden units, n_h
#  - num training examples, m
#  - minibatch size bs
#  - variance of gaussian for initialization of a weights
#
# train method returns:
#  - w_ijs - ndarray of w_ij for each grad descent step.
#  - a, b  - values of biases after training
#
# TODO:
#  - add monitor of cost function
#  - implement momentum, RMSProp, Learning rate decay as independent modules
#    that don't add to the complexity of the code (so that turning them on
#    doesn't change form/readability of code.)
#  - w_ij is currently implemented as (n_v x n_h). Change to (n_h x n_v)
#    matrix to conform to Ng's notation.


class RBM:
  def __init__(self, numVis, numHid, numTrain, bs):
    '''
    Args:
      numVis: number of visible units
      numHid: number of hidden units
      numTrain: number of training examples
      bs: minibatch size
    '''
    self.n_v = numVis
    self.n_h = numHid
    self.m = numTrain
    self.bs = bs
    self.trained = False

  def setParams(self, W, a, b):
    '''
    Sets RBM parameters

    Args:
      W: (n_v x n_h) ndarray of trained couplings
      a: (n_v x 1) ndarray of trained visible biases
      b: (n_h x 1) ndarray of trained hidden biases

    Returns:
    '''
    self.w_ij, self.a, self.b = W, a, b
    self.trained = True 

  def train(self, data, numEpochs, trainRate, biasesTo0=False, allParams=False,
            l1RegWeight=0, momentum=0, log_interval=10):
    '''
    Trains the RBM

    Args:
      data: (n_v x m) ndarray of inputs
      numEpochs: number of epochs to train for
      trainRate: training rate
      biasesTo0: if true, biases are left out of calculation
      allParams: if true, log a, b, and wij every "log_interval"
      l1RegWeight: strength of L1 regularization (no reg if 0)
      momentum: strength of momentum (no mom if 0)
      log_interval: log progress every this number of epochs

    Returns:
      w_ijs: list of (n_v x n_h) ndarrays of trained couplings
      aa: list of(n_v x 1) ndarrays of trained visible biases
      bb: list of (n_h x 1) ndarrays of trained hidden biases
    '''
    # make sure data is of specified shape
    assert (data.shape[0] == self.n_v)
    # number of samples must be integer multiple of bs
    assert (np.mod(data.shape[1], self.bs) == 0)

    batchesInSample = self.m / self.bs

    # Init biases
    if biasesTo0 is True:
      self.a, self.b = np.zeros((self.n_v, 1)), np.zeros((self.n_h, 1))
    else:
      # fraction of samples with i'th spin on (HINTON section 8)
      vp = 1. * np.sum(data, axis=1, keepdims=True) / self.m
      self.a, self.b = np.log(vp / (1. - vp)), np.ones((self.n_h, 1))

    # initialize weights to gaussian small values (HINTON)
    np_rng = np.random.RandomState(1234)
    self.w_ij = np_rng.normal(0, 0.01, size=(self.n_v, self.n_h))

    # Placeholder for momentum
    v = np.zeros(self.w_ij.shape)

    # For all w_ijs, as, and bs in run
    w_ijs, aa, bb = [], [], []

    # This is the train routine. At the end of this loop, we should have an list
    # of w_ij matrices -- one for every gradient ascent step.
    for i in tqdm(range(numEpochs)):
      # randomize sample order for this epoch
      dataThisEpoch = np.copy(data)
      np.random.shuffle(dataThisEpoch.T)

      # make batches for this epoch
      batches = np.split(dataThisEpoch, batchesInSample, axis=1)

      for batch in batches:
        # probability that hidden unit is 1
        # Gives (n_h x bs) matrix
        pHidData = self._logistic(np.dot(self.w_ij.T, batch) + self.b)

        # draw a sample from pHidData
        sampHidData = np.random.binomial(1, pHidData)

        # reconstructed visible pdf from the hidden data sample
        pVisRecon = self._logistic(np.dot(self.w_ij, sampHidData) + self.a)

        # sample of this pdf
        sampVisRecon = np.random.binomial(1, pVisRecon)

        # reconstructed hidden pdf
        pHidRecon = self._logistic(np.dot(self.w_ij.T, pVisRecon) + self.b)

        # <v h> correlations for data and reconstructed
        visHidCorrData = (1. / self.bs) * np.dot(batch, pHidData.T)
        visHidCorrRecon = (1. / self.bs) * np.dot(pVisRecon, pHidRecon.T)

        # gradient ascent on parameters, with opt L1 regularization
        # TODO check minus sign
        v = momentum * v + trainRate * (visHidCorrData - visHidCorrRecon -
                                        l1RegWeight * np.sign(self.w_ij))
        self.w_ij += v
        if biasesTo0 is False:
          self.a += (trainRate / self.bs) * np.sum(batch - pVisRecon,
                                                   axis=1, keepdims=True)
          self.b += (trainRate / self.bs) * np.sum(pHidData - pHidRecon,
                                                   axis=1, keepdims=True)

      # log weights during training if 'allParams' is set
      if allParams == True and i % log_interval == 0:
        w_ijs.append(self.w_ij)
        aa.append(self.a)
        bb.append(self.b)

    # final result
    w_ijs.append(self.w_ij)
    aa.append(self.a)
    bb.append(self.b)

    self.trained = True
    return w_ijs, aa, bb

  def vToh(self, vis):
    assert (self.trained == True)
    assert (vis.shape[0] == self.n_v)

    # Calculate final hidden activations from final model
    return self._logistic(np.dot(self.w_ij.T, vis) + self.b)

  def hTov(self, hid):
    assert (self.trained == True)
    assert (hid.shape[0] == self.n_h)

    # Figure out what biases to put in here (compare vToH function, above)
    return self._logistic(np.dot(self.w_ij, hid))

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

  def _tanh(self, x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
