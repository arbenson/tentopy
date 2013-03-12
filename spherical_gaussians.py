#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time
import math

def compute_data(M2, M3, w, A):
  W, X3 = la.whiten(M2, M3)
  evals, evecs = la.reconstruct(W, X3)

  return ([abs(w[i] - evals[i]) / abs(w[i]) for i in xrange(len(w))],
          [np.linalg.norm(evecs[i,:] - A[:, i], 2) / np.linalg.norm(evecs[i,:], 2) \
               for i in xrange(A.shape[1])])
    
class SGDataGen:
  def __init__(self, w, A, sigma2, data_interval=1000, cov_perturb=None, noise=None):
    self.w = w
    self.cumw = np.cumsum(w)
    self.A = A
    self.d = A.shape[0]
    self.sigma2 = sigma2
    self.cov = sigma2 * np.eye(self.d)
    self.data_interval = data_interval
    if cov_perturb != None:
      self.cov += cov_perturb
    self.noise = noise

    self.eval_errs = []
    self.evec_errs = []

  # TODO: combine with equivalent form from other data generators
  def index_sample(self):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(self.cumw):
      if unif_samp <= s:
        return i
    
    return len(self.cumw) - 1

  def gen(self, n):
    for k in xrange(n):
      i = self.index_sample()
      sample = self.A[:, i] + rand.multivariate_normal(np.zeros(self.d), self.cov)
      if self.noise != None:
        sample += self.noise(None)
      yield sample

  def compute_M2_M3(self, cross, M2, samples, n):
    cov = np.cov(np.array(samples).T)
    mean = np.mean(samples, axis=0)
    cross /= n
    
    T = la.tensor_outer(np.zeros(self.d), 3)
    for i in xrange(self.d):
      T[:, i, i] += mean
      T[i, :, i] += mean
      T[i, i, :] += mean
  
    eigs, _ = np.linalg.eig(cov)
    lambda_min = eigs[-1]
    return M2 / n - lambda_min * np.eye(self.d), cross - lambda_min * T

  def inner_gen_tensor(self, n):
    samples = []
    M2 = np.zeros((self.d, self.d))
    cross = la.tensor_outer(np.zeros(self.d), 3)

    for i, sample in enumerate(self.gen(n)):
      samples.append(sample)
      cross += la.tensor_outer(sample, 3)
      M2 += np.outer(sample, sample)
      if (i+1) % self.data_interval == 0:
        print "\n computing data... %d" % (i+1)
        M2, M3 = self.compute_M2_M3(np.copy(cross), np.copy(M2), samples, n)
        eval_err, evec_err = compute_data(M2, M3, self.w, A)
        self.eval_errs.append(eval_err)
        self.evec_errs.append(evec_err)

    M2, M3 = self.compute_M2_M3(np.copy(cross), np.copy(M2), samples, n)
    return M2, M3

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)


if __name__ == "__main__":
  w = [0.6, 0.3, 0.1]
  A = np.array([[3.0, 5.0, 2.0], [1.0, 3.75, 4.0], [2.5, 0.5, 1.25]])

  N = 10000
  sigma2 = 2.

  eval = []
  evec = []
  
  for eps in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1, 2, 3, 4]:
    eval_sum = np.zeros(len(w))
    evec_sum = np.zeros(len(w))
    num_trials = 12
    for k in xrange(num_trials):
      cov_perturb = np.zeros((A.shape[0], A.shape[0]))
      cov_perturb[0][0] += eps
      cov_perturb[1][1] += 2 * eps
      cov_perturb[0][1] = eps
      cov_perturb[1][0] = eps
      #cov_perturb = None
  
      exponential = lambda x: eps * np.random.exponential(scale=1., size=A.shape[0])
      #exponential = None
  
      g = SGDataGen(w, A, sigma2, data_interval=N, cov_perturb=cov_perturb,
                    noise=exponential)
      g.gen_tensor(N)
      
      eval_sum += g.eval_errs[0]
      evec_sum += g.evec_errs[0]
  
    eval.append(list(eval_sum / num_trials))
    evec.append(list(evec_sum / num_trials))
  
  print eval
  print evec

