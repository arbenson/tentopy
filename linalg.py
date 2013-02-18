#!/usr/bin/env/python
"""
linalg.py

Linear algebra tools for orthogonal tensor decompositions.
"""

import numpy as np

"""
Compute the approximate eigenvalue corresponding to an approximate eigenvector

inputs:
T: a super symmetric tensor
v: an approximate eigenvalue
"""
def approx_eval(T, v):
  return np.dot(np.tensordot(np.dot(T, v), v), v)

"""
Compute a rank-1 order-3 tensor computation by outer products.

input: v the basis vector
output: \otimes^3 v
"""
def tensor_outer(v):
  # TODO: Make this rank-1 computation faster.  It is not clear how to do this
  # with native numpy operations.
  # TODO: generalize this to greater than order 3
  X = np.outer(v, v)
  return np.array([v[i] * X for i in xrange(len(v))])

"""
Main power method driver.  Computes the (eigenvalue, eigenvector) pair of a
tensor corresponding to the largest eigenvalue.

This algorithm is based on the robust tensor power method, described in
Algorithm 1 of "Tensor Decompositions for Learning Latent Variable Models"
by Anandkumar et al.

TODO:
  - support arbitrary-order tensors (currently only support for order 3)

inputs:
T: a super-symmetric tensor
L: number of inner iterations of power method to perform
N: number of iterations per inner iteration

outputs:
a tuple:
    (approximate largest eigenvalue, corresponding eigenvector, deflated tensor)
"""
def power_method(T, L, N):
  # TODO: extend to support order-n tensors instead of just order-3
  if len(T.shape) != 3:
    raise Exception('Only supporting order-3 tensors for now')
  if not (T.shape[0] == T.shape[1] == T.shape[2]):
    raise Exception('Each tensor dimension must be the same')
  k = T.shape[0]
  thetas = []

  def inner_iter(N, theta):
    for t in xrange(N):
      next_iter = theta * np.dot(np.tensordot(T, np.eye(k)).reshape(1, k),
                                 theta)
      theta = next_iter / np.linalg.norm(next_iter, 2)
    return theta

  for tau in xrange(L):
    # Choose a starting vector unfiormly at random from unit ball
    v = np.random.randn(k)
    theta_0 = v / np.linalg.norm(v, 2)
    theta_0 = theta_0.reshape((k, 1))
    thetas.append(inner_iter(N, theta_0))

  ind = np.argmax([approx_eval(T, theta) for theta in thetas])
  theta_hat = inner_iter(N, thetas[ind])
  lambda_hat = approx_eval(T, theta_hat)
  rank1_approx = lambda_hat * tensor_outer(theta_hat)

  return theta_hat, lambda_hat, T - rank1_approx

if __name__ == '__main__':
  k = 4
  eigs = 50 * np.random.randn(k)
  v = np.zeros(4)
  T = tensor_outer(v)
  for i in xrange(4):
    v = np.random.randn(k)
    v = v / np.linalg.norm(v, 2)
    T += eigs[i] * tensor_outer(v)
  print max(eigs)
  theta_hat, lambda_hat, dT = power_method(T, 1000, 10)
  print lambda_hat
