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
    for p in xrange(n):
      i, j, k = self.triple_draw()
      cross[i, j, k] += 1
    cross /= n
    return cross

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)

w = [0.7, 0.3]
A = np.array([[0.5, 0.2], [0.0, 0.8], [0.5, 0.], [0.0, 0.]])
g = ESTMDataGen(w, A)
print A

t0 = time.time()
M3 = g.gen_tensor(1000000)
evecs, evals = la.eig(M3, 25, 200)
# normalized eigenvalues
print evals[0:2] / sum(evals[0:2])
for k in xrange(2):
  print evecs[k, :].T / np.linalg.norm(evecs[k, :], 1)
print "time: ", time.time() - t0