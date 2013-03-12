#!/usr/bin/env python

import matplotlib.pyplot as plt

def plot_data(x, y, title, data_type, xlabel='$\epsilon$', leg_loc=2):
  plt.figure()
  plt.loglog(x, y)
  plt.title(title)
  plt.xlabel(xlabel)
  if data_type == 'evals':
    plt.ylabel('relative error in $w_i$ approximation')
    plt.legend(['$|\hat{w}_%d - w_%d| / |w_%d|$' % (i, i, i) for i in xrange(len(y[0]))], loc=leg_loc)
  elif data_type == 'evecs':
    plt.ylabel('relative error in $\mu_i$ approximation')
    plt.legend(['$||\hat{\mu}_%d - \mu_%d||_2 / ||\mu_%d||_2$' % (i, i, i) for i in xrange(len(y[0]))], loc=leg_loc)
  plt.show()





