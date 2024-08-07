% number of repeats:% 3
% enter first, last, inc:% 48 480 48 
data = [
%  n          reference      |         current implementation 
%        time       GFLOPS   |    time       GFLOPS     diff 
   480 6.5546e-03 3.3745e+01    2.7632e-02 8.0046e+00 3.5527e-13
   432 3.4363e-03 4.6924e+01    2.1076e-02 7.6505e+00 3.1264e-13
   384 2.3641e-03 4.7903e+01    1.4619e-02 7.7468e+00 1.9895e-13
   336 1.5465e-03 4.9056e+01    9.1057e-03 8.3317e+00 1.7053e-13
   288 1.0596e-03 4.5090e+01    6.0718e-03 7.8685e+00 1.1369e-13
   240 6.5346e-04 4.2310e+01    3.3840e-03 8.1703e+00 4.2633e-14
   192 2.9640e-04 4.7760e+01    1.7810e-03 7.9480e+00 2.8422e-14
   144 1.5659e-04 3.8139e+01    6.1173e-04 9.7624e+00 2.8422e-14
    96 4.3975e-05 4.0238e+01    1.8454e-04 9.5885e+00 1.4211e-14
    48 9.2070e-06 2.4023e+01    2.3548e-05 9.3929e+00 7.1054e-15
];

% Maximum difference between reference and your implementation: 3.552714e-13.
