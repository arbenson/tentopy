#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time

class ESTMDataGen:
  def __init__(self, w, A):
    self.cumw = np.cumsum(w)
    self.A = A
    self.cumA = np.cumsum(A, 0)
    self.d = A.shape[0]

  def index_sample(self):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(self.cumw):
      if unif_samp <= s:
        return i
    
    return len(self.cumw) - 1

  # TODO: consolidate with index_sample()
  def doc_sample(self, j):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(self.cumA[:,j]):
      if unif_samp <= s:
        return i
    
    return len(self.cumw) - 1

  def triple_draw(self):
    j = self.index_sample()
    return [self.doc_sample(j) for k in xrange(3)]

  def inner_gen_tensor(self, n):
    cross = la.tensor_outer(np.zeros(self.d), 3)
    M2 = np.zeros((self.d, self.d))
    for p in xrange(n):
      i, j, k = self.triple_draw()
      cross[i, j, k] += 1.
      M2[i, j] += 1
    M2 /= n
    cross /= n
    return M2, cross

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)

if __name__ == "__main__":
  w = [0.5, 0.3, 0.2]
  A = np.array([[0.6, 0.3, 0.2], [0.3, 0.2, 0.5], [0.2, 0.5, 0.3]])
  #print "original weights:\n", w
  #print "original distribs:\n", A

  eval_errs = []
  evec_errs = []
  conds = []

  N = 100000

  for eps in [1e-1, 1 - 1e-2, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8, 1 - 1e-10, 1 - 1e-12]:
    B = np.zeros((3, 3))
    B[:, 0:2] += A[:, 0:2]
    B[:, 2] += eps * A[:, 1] + (1 - eps) * A[:, 2]
    conds.append(np.linalg.cond(B, 2))

    num_trials = 12
    eval_trial_err = np.zeros(len(w))
    evec_trial_err = np.zeros(B.shape[1])

    for trial in xrange(num_trials):
      g = ESTMDataGen(w, B)
      M2, M3 = g.gen_tensor(N)
      W, X3 = la.whiten(M2, M3)
      evals, evecs = la.reconstruct(W, X3)

      eval_trial_err += np.array([abs(w[i] - evals[i]) / abs(w[i]) for i in xrange(len(w))])
      evec_trial_err += np.array([np.linalg.norm(evecs[i,:] - B[:, i], 2) / np.linalg.norm(evecs[i,:], 2) \
                                  for i in xrange(B.shape[1])])

    print "updating..."
    eval_errs.append(list(eval_trial_err / num_trials))
    evec_errs.append(list(evec_trial_err / num_trials))

  print conds
  print eval_errs
  print evec_errs

