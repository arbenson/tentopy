#!/usr/bin/env/python
"""
linalg.py

Linear algebra tools for orthogonal tensor decompositions.

The power method algorithm is based on the robust tensor power method, described
in Algorithm 1 of "Tensor Decompositions for Learning Latent Variable Models"
by Anandkumar et al.
"""

import numpy as np

def approx_eval(T, v):
  """ Compute the approximate eigenvalue corresponding to an approximate
  eigenvector.
  
  inputs:
  T: a super symmetric tensor
  v: an approximate eigenvalue
  
  output:
   approximate eigenvalue T(v, v, v)
  """
  return np.dot(np.tensordot(T, np.outer(v, v)), v)

def tensor_outer(v, n):
  """
  Compute a rank-1 order-n (n > 1) tensor computation by outer products.
  
  input: v the basis vector
  output: \otimes^3 v
  """
  T = np.outer(v, v)
  for i in xrange(n - 2):
    T = np.outer(v, T)
  return T.reshape([len(v)] * n)

def power_method(T, L, N):
  """ Main power method driver.  Computes the (eigenvalue, eigenvector) pair
  of a tensor corresponding to the largest eigenvalue.
  
  TODO:
    - support arbitrary-order tensors (currently only support for order 3)
  
  inputs:
  T: a super-symmetric tensor
  L: number of inner iterations of power method to perform
  N: number of iterations per inner iteration
  
  outputs:
  a tuple:
      (approximate largest eigenvalue, corresponding eigenvector,
       deflated tensor)
  """
  k = T.shape[0]
  thetas = []

  def inner_iter(N, theta):
    for t in xrange(N):
      next_iter = np.tensordot(T, np.outer(theta, theta))
      theta = next_iter / np.linalg.norm(next_iter, 2)
    return theta.T

  for tau in xrange(L):
    # Choose a starting vector unfiormly at random from unit ball
    v = np.random.randn(k)
    theta_0 = v / np.linalg.norm(v, 2)
    theta_0 = theta_0.reshape((k, 1))
    thetas.append(inner_iter(N, theta_0))

  ind = np.argmax([approx_eval(T, theta) for theta in thetas])
  theta_hat = inner_iter(N, thetas[ind])
  lambda_hat = approx_eval(T, theta_hat)
  rank1_approx = lambda_hat * tensor_outer(theta_hat, 3)

  return theta_hat, lambda_hat, T - rank1_approx

def eig(T, L=10, N=10):
  """ Compute the eigen-decomposition of a super-symmetric tensor.

  inputs:
  T: a super-symmetric tensor
  L: number of inner iterations of power method to perform
  N: number of iterations per inner iteration

  outputs:
  a tuple of eigenvectors and eigenvalues
  """
  # TODO: extend to support order-n tensors instead of just order-3
  if len(T.shape) != 3:
    raise Exception('Only supporting order-3 tensors for now')
  if not (T.shape[0] == T.shape[1] == T.shape[2]):
    raise Exception('Each tensor dimension must be the same')
  k = T.shape[0]
  evecs = []
  evals = []
  for i in xrange(k):
    evec, eval, def_T = power_method(T if i == 0 else def_T, L, N)
    evecs.append(list(evec))
    evals.append(eval)
  return np.array(evecs), evals

if __name__ == '__main__':
  T = tensor_outer(np.zeros(4), 3)
  T[0][0][0] = 100
  T[1][1][1] = 200
  T[2][2][2] = 50
  T[3][3][3] = 75
  print eig(T)
