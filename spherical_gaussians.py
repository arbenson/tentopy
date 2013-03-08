#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la

class SGDataGen:
  def __init__(self, w, A, sigma2):
    self.cumw = np.cumsum(w)
    self.A = A
    self.d = A.shape[0]
    self.cov = sigma2 * np.eye(self.d)

  def index_sample(self):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(self.cumw):
      if unif_samp <= s:
        return i
    
    return len(self.cumw) - 1

  def gen(self, n):
    for k in xrange(n):
      i = self.index_sample()
      yield self.A[:, i] + rand.multivariate_normal(np.zeros(self.d), self.cov)

  def gen_tensor(self, n):
    samples = []
    cross = la.tensor_outer(np.zeros(self.d), 3)
    for sample in self.gen(n):
      samples.append(sample)
      cross += la.tensor_outer(sample, 3)
    cov = np.cov(np.array(samples).T)
    mean = np.mean(samples, axis=0)
    cross /= n

    T = la.tensor_outer(np.zeros(self.d), 3)
    for i in xrange(self.d):
      ei = np.eye(self.d)[i,:]
      T += la.tensor_outer2(mean, ei, ei)
      T += la.tensor_outer2(ei, mean, ei)
      T += la.tensor_outer2(ei, ei, mean)

    eigs, _ = np.linalg.eig(cov)
    lambda_min = eigs[-1]
    return cross - lambda_min * T

w = [0.75, 0.25]
#A = rand.random((4, 2))
A = np.eye(4)[:,0:2]
sigma2 = 2
g = SGDataGen(w, A, sigma2)

M3 = g.gen_tensor(100000)
print A
evecs, evals = la.eig(M3, 100, 100)
print evals[0:2]
print evecs[0:2,:].T

