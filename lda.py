#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time
import math

class LDADataGen:
  def __init__(self, alpha, A, noise=None):
    self.alpha = alpha
    self.alpha_0 = sum(alpha)
    self.A = A
    self.cumA = np.cumsum(A, 0)
    self.d = A.shape[0]
    self.noise = noise

  def doc_sample(self, j):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(self.cumA[:,j]):
      if unif_samp <= s:
        return i
    
    print "possible problem"
    return len(self.cumw) - 1

  # TODO: combine with doc_sample() and tools from estm
  def get_doc(self, h):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(np.cumsum(h)):
      if unif_samp <= s:
        return i

    print "possible problem"    
    return len(np.cumsum(h)) - 1

  def triple_draw(self):
    h = rand.dirichlet(self.alpha)
    if self.noise != None:
      h += self.noise(None)
      h = h / sum(h)
    return [self.doc_sample(self.get_doc(h)) for i in xrange(3)]

  def compute_M3(self, cross, M1, M2, x12_data, n):
    c = (2 * (self.alpha_0 ** 2)) / ((self.alpha_0 + 2) * (self.alpha_0 + 1))
    M3 = cross + c * la.tensor_outer(M1, 3)
 
    T = la.tensor_outer(np.zeros(self.d), 3)
    for i, j in x12_data:
      T[i, j, :] += M1
      T[i, :, j] += M1
      T[:, i, j] += M1
    T /= n

    M2 -= (self.alpha_0 / (self.alpha_0 + 1)) * np.outer(M1, M1)
    M3 -= (self.alpha_0 / (self.alpha_0 + 2)) * T
    return M2, M3

  def inner_gen_tensor(self, n):
    cross = la.tensor_outer(np.zeros(self.d), 3)
    M1 = np.zeros(self.d)
    M2 = np.outer(np.zeros(self.d), np.zeros(self.d))
    x12_data = []

    for p in xrange(n):
      i, j, k = self.triple_draw()
      M1[i] += 1.
      M2[i, j] += 1.
      cross[i, j, k] += 1.
      x12_data.append((i, j))

    M1 /= n
    M2 /= n
    cross /= n

    return self.compute_M3(np.copy(cross), np.copy(M1), np.copy(M2),
                           x12_data, n)

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)


if __name__ == "__main__":

  alpha = [0.4, 0.2, 0.1, 0.05]
  alpha_0 = sum(alpha)
  d = 4
  A = np.eye(d)
  A[0, 0] = 0.5
  A[-1, 0] = 0.5
  A[0,1] = 0.25
  A[1,1] = 0.25
  A[2,1] = 0.5

  eval_errs = []
  evec_errs = []

  for eps in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1, 2, 3, 4]:
    eval_trial_err = np.zeros(d)
    evec_trial_err = np.zeros(d)
    num_trials = 12
    for k in xrange(num_trials):
      noise = lambda x: eps * abs(np.random.multivariate_normal(np.zeros(d), np.eye(d)))
      g = LDADataGen(alpha, A, noise=noise)
      N = 20000
      M2, M3 = g.gen_tensor(N)
      W, X3 = la.whiten(M2, M3)
      evals, evecs = la.reconstruct(W, X3)
      # reconstruct the alpha_i
      comp_alpha = evals * alpha_0 / sum(evals)

      eval_trial_err += np.array([abs(alpha[i] - comp_alpha[i]) / abs(alpha[i]) \
                                      for i in xrange(len(alpha))])
      evec_trial_err += np.array([np.linalg.norm(evecs[i,:] - A[:, i], 2) / \
                                  np.linalg.norm(A[:,i], 2) \
                                      for i in xrange(A.shape[1])])

    print "updating..."
    eval_errs.append(list(eval_trial_err / num_trials))
    evec_errs.append(list(evec_trial_err / num_trials))

  print eval_errs
  print evec_errs
