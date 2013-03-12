#!/usr/bin/env python

import matplotlib.pyplot as plt
import plotting_tools as pt

# N = 100000 samples, avg. over 12 trials
conds = [4.3587805434989768, 446.54297500359473, 44769.389497660537, 4477054.8049021363, 447705591.98851293, 44770564424.867004, 4477451048236.0947]
evals = [[0.036347142225399309, 0.077615635365627281, 0.22101634594016176], [18.55852278587513, 9.4829219517486791, 1.0012458530835742], [36.139855533220945, 4.5310572238528461, 0.98099950388454216], [35.50560767315617, 3.820212655256825, 0.94081456531366692], [22.319542344319185, 7.2068766793380261, 0.9099779793015661], [64.417199748732983, 3.5425327530586763, 1.0449281296847499], [1909.0496500216384, 9.5226461391681614, 1.0132024120636247]]
evecs = [[0.15221056657220564, 0.089887096150279941, 0.14145227108019071], [34.56032999904815, 14.73674468794338, 1.102940404559813], [44.171861295215706, 21.228442501283144, 1.0183389454200404], [44.09841401218398, 12.001030577761165, 1.0706352221096669], [49.082707147959887, 15.964960822927607, 1.1625858984940112], [56.20387070769474, 13.841156772618405, 0.98333703786477944], [88.797863573706039, 6.6012981841031868, 1.2596075536603129]]

for data, data_type, leg_loc in [(evals, 'evals', 2), (evecs, 'evecs', 4)]:
  pt.plot_data(conds, data, '100,000 samples, 12 trials', data_type,
               xlabel='$\kappa_2(A)$', leg_loc=leg_loc)
