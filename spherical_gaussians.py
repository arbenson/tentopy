#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time
import math

def compute_data(M2, M3, w, A):
  evecs, evals = la.eig(M3, 20, 200)
  evals = evals / sum(evals)
  eval_error1 = [abs(w[i] - evals[i]) / abs(w[i]) for i in xrange(len(w))]
  evec_error1 = [min(np.linalg.norm(A[:, i] - evecs[i, :], 2),
                  np.linalg.norm(A[:, i] + evecs[i, :], 2))  for i in xrange(len(w))]

  W, X3 = la.whiten(M2, M3)
  evecs, evals = la.eig(X3, 20, 200)
  eval_real = np.array([(1. / eval) ** 2 for eval in evals])
  eval_real = eval_real / sum(eval_real)
  eval_real = eval_real[::-1]

  eval_error2 = [abs(w[i] - eval_real[i]) / abs(w[i]) for i in xrange(len(w))]
  evec_error2 = []

  for i in xrange(len(w)):
    u = evecs[i, :]
    u = np.linalg.solve(W.T, u) * evals[i]
    exp_u = A[:, len(w) - i - 1]
    evec_error2.append(min(np.linalg.norm(exp_u - u, 2), np.linalg.norm(exp_u + u, 2)))

  return eval_error1, evec_error1, eval_error2, evec_error2


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

    self.eval_errors1 = []
    self.evec_errors1 = []
    self.eval_errors2 = []
    self.evec_errors2 = []

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
      ei = np.eye(self.d)[i,:]
      T += la.tensor_outer2(mean, ei, ei)
      T += la.tensor_outer2(ei, mean, ei)
      T += la.tensor_outer2(ei, ei, mean)
  
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
        eval_error1, evec_error1, eval_error2, evec_error2 = compute_data(M2, M3, self.w, A)
        self.eval_errors1.append(eval_error1)
        self.evec_errors1.append(evec_error1)
        self.eval_errors2.append(eval_error2)
        self.evec_errors2.append(evec_error2)

    M2, M3 = self.compute_M2_M3(np.copy(cross), np.copy(M2), samples, n)
    return M2, M3

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)


w = [0.75, 0.25]
A = np.eye(3)[:, 0:2]
sigma2 = 4.

eval1 = []
evec1 = []
eval2 = []
evec2 = []

for eps in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1, 2, 3]:
  cov_perturb = np.zeros((A.shape[0], A.shape[0]))
  cov_perturb[0][0] += eps
  cov_perturb[1][1] -= eps
  cov_perturb[2][2] += 2 * eps
  cov_perturb[0][1] = eps
  cov_perturb[1][0] = eps
  #cov_perturb = None

  gamma = lambda x: eps * np.random.gamma(2., 2., A.shape[0])
  #gamma = None

  N = 200000
  g = SGDataGen(w, A, sigma2, data_interval=N, cov_perturb=cov_perturb,
                noise=gamma)
  g.gen_tensor(N)
  #print g.eval_errors1
  #print g.evec_errors2
  #print g.eval_errors1
  #print g.evec_errors2
  
  eval1.append(g.eval_errors1[0])
  evec1.append(g.evec_errors1[0])
  eval2.append(g.eval_errors2[0])
  evec2.append(g.evec_errors2[0])

print eval1
print evec1
print eval2
print evec2
