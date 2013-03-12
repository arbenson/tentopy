#!/usr/bin/env python

import numpy as np
import numpy.random as rand
import linalg as la
import time
import math

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

  # TODO: combine with doc_sample() and tools from estm
  def get_doc(self, h):
    unif_samp = rand.uniform(0, 1)
    for i, s in enumerate(np.cumsum(h)):
      if unif_samp <= s:
        return i
    
    return len(np.cumsum(h)) - 1

  def triple_draw(self):
    h = rand.dirichlet(self.alpha)
    j = self.get_doc(h)
    return [self.doc_sample(j) for k in xrange(3)]

  def compute_M3(self, cross, M1, x12_data, n):
    c = (2 * self.alpha_0 ** 2) / ((self.alpha_0 + 2) * (self.alpha_0 + 1))
    M3 = cross / n + c * la.tensor_outer(M1, 3)
 
    M2 = np.outer(np.zeros(self.d), np.zeros(self.d))
    T = la.tensor_outer(np.zeros(self.d), 3)
    for i, j in x12_data:
      M2[i, j] += 1.
      T[i, j, :] += M1
      T[i, :, j] += M1
      T[:, i, j] += M1
    T /= n

    M2 = M2 / n - (self.alpha_0 / (self.alpha_0 + 1)) * np.outer(M1, M1)
    M3 -= (self.alpha_0 / (self.alpha_0 + 2)) * T
    return M2, M3

  def inner_gen_tensor(self, n):
    cross = la.tensor_outer(np.zeros(self.d), 3)
    M1 = np.zeros(self.d)
    x12_data = []

    for p in xrange(n):
      i, j, k = self.triple_draw()
      cross[i, j, k] += 1.
      M1[i] += 1.
      x12_data.append((i, j))

    M1 /= n

    return self.compute_M3(np.copy(cross), np.copy(M1),
                           x12_data, n)

  def gen_tensor(self, n):
    return self.inner_gen_tensor(n)



if __name__ == "__main__":
  alpha = [1., 5., 8.]
  alpha_0 = sum(alpha)
  A = np.array([[0.6, 0.3, 0.2], [0.3, 0.2, 0.5], [0.2, 0.5, 0.3]])
  g = LDADataGen(alpha, A)
  
  M2, M3 = g.gen_tensor(50000)
  
  W, X3 = la.whiten(M2, M3)
  evals, evecs = la.reconstruct(W, X3)
  print evals
  print evecs

