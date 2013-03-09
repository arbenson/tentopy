#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time

class LDADataGen:
  def __init__(self, alpha, A):
    self.alpha = alpha
    self.alpha_0 = sum(alpha)
    self.A = A
    self.cumA = np.cumsum(A, 0)
    self.d = A.shape[0]

  def doc_sample(self, j):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(self.cumA[:,j]):
      if unif_samp <= s:
        return i
    
    return len(self.cumw) - 1

  def get_doc(self, h):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(np.cumsum(h)):
      if unif_samp <= s:
        return i
    
    return len(self.cumw) - 1

  def triple_draw(self):
    h = rand.dirichlet(self.alpha)
    j = self.get_doc(h)
    return [self.doc_sample(j) for k in xrange(3)]

  def compute_M3(self, cross, M1, x12_data, n):
    M3 = la.tensor_outer(np.zeros(self.d), 3)
    M3 += cross / n
    c = (2 * self.alpha_0 ** 2) / ((self.alpha_0 + 2) * (self.alpha_0 + 1))
    M3 += c * la.tensor_outer(M1 / n, 3)
 
    T = la.tensor_outer(np.zeros(self.d), 3)
    for i, j in x12_data:
      x1 = np.eye(self.d)[i,:]
      x2 = np.eye(self.d)[j,:]
      T += la.tensor_outer2(x1, x2, M1)
      T += la.tensor_outer2(x1, M1, x2)
      T += la.tensor_outer2(M1, x1, x2)

    T /= n
    M3 -= T * self.alpha_0 / (self.alpha_0 + 2)
    return M3

  def inner_gen_tensor(self, n):
    cross = la.tensor_outer(np.zeros(self.d), 3)
    M1 = np.zeros(self.d)
    x12_data = []

    for p in xrange(n):
      i, j, k = self.triple_draw()
      cross[i, j, k] += 1.
      M1[i] += 1.
      x12_data.append((i, j))

    return self.compute_M3(cross, M1, x12_data, n)

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)


alpha = [1, 25, 30]
A = np.eye(4)[:, 0:3]
g = LDADataGen(alpha, A)

t0 = time.time()
M3 = g.gen_tensor(100000)
evecs, evals = la.eig(M3, 25, 200)
# normalized eigenvalues
print evals[0:3] / sum(evals[0:3])
for k in xrange(3):
  print evecs[k, :].T / np.linalg.norm(evecs[k, :], 1)
print "time: ", time.time() - t0
