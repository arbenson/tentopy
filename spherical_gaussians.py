#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time

class SGDataGen:
  def __init__(self, w, A, sigma2):
    self.cumw = np.cumsum(w)
    self.A = A
    self.d = A.shape[0]
    self.sigma2 = sigma2
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
      #rand.gamma(shape=2., scale=1., size=self.d)
      yield self.A[:, i] + rand.multivariate_normal(np.zeros(self.d), self.cov)

  def compute_M3(self, cross, samples, n):
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

  def inner_gen_tensor(self, n):
    samples = []
    cross = la.tensor_outer(np.zeros(self.d), 3)

    for i, sample in enumerate(self.gen(n)):
      samples.append(sample)
      cross += la.tensor_outer(sample, 3)
      #if not (i+1 % 1000):
      #  M3 = self.compute_M3(cross, samples, n)
      #  evecs, evals = la.eig(M3, 100, 100)
      #  print abs(evals[0] - 0.75)

    return self.compute_M3(cross, samples, n)

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)
    

w = [0.75, 0.25]
#A = np.eye(4)[:,0:2]
A = np.array([[0.3, 0.5], [0.2, 0.1], [0.3, 0.2], [0.2, 0.2]])
print A
sigma2 = 2.
g = SGDataGen(w, A, sigma2)

t0 = time.time()
M3 = g.gen_tensor(100000)
evecs, evals = la.eig(M3, 25, 300)
# normalized eigenvalues
print evals[0:2] / sum(evals[0:2])
for k in xrange(2):
  print evecs[k,:] / np.linalg.norm(evecs[k,:], 1)
print "time: ", time.time() - t0
