#!/usr/bin/env python

import linalg
import numpy as np

N = 5
T = linalg.tensor_outer(np.random.rand(N), 3)
theta = np.random.rand(N)

comp = np.tensordot(T, np.outer(theta, theta))
comp_test = np.zeros(N)
for k in xrange(N):
  for j2 in xrange(N):
    for j3 in xrange(N):
      comp_test[k] += T[k, j2, j3] * theta[j2] * theta[j3]

# should be the same
print comp, comp_test

eval = linalg.approx_eval(T, theta)
eval_test = 0
for j1 in xrange(N):
  for j2 in xrange(N):
    for j3 in xrange(N):
      eval_test += T[j1, j2, j3] * theta[j1] * theta[j2] * theta[j3]

# should be the same
print eval, eval_test

N = 3
u = np.random.rand(N)
v = np.random.rand(N)
w = np.random.rand(N)
M3 = np.outer(u, np.outer(v, w))
M3 = M3.reshape([N] * 3)

X3 = linalg.tensor_outer(np.zeros(N), 3)
x = np.random.rand(N)
y = np.random.rand(N)
W = np.outer(x, y)

for i1 in xrange(N):
  for i2 in xrange(N):
    for i3 in xrange(N):
      for j1 in xrange(N):
        for j2 in xrange(N):
          for j3 in xrange(N):
            X3[i1, i2, i3] += M3[j1, j2, j3] * W[j1, i1] * W[j2, i2] * W[j3, i3]

Y3 = np.tensordot(M3, ()

print "whitening: "
print X3
print Y3
      

