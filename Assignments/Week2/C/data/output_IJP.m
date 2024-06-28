% number of repeats:% 3
% enter first, last, inc:% 48 480 48 
data = [
%  n          reference      |         current implementation 
%        time       GFLOPS   |    time       GFLOPS     diff 
   480 9.6496e-03 2.2921e+01    1.4676e-01 1.5071e+00 3.5527e-13
   432 4.0582e-03 3.9732e+01    1.0490e-01 1.5372e+00 3.1264e-13
   384 3.1243e-03 3.6246e+01    7.7425e-02 1.4627e+00 1.9895e-13
   336 2.0877e-03 3.6339e+01    4.9562e-02 1.5307e+00 1.7053e-13
   288 1.1848e-03 4.0325e+01    3.0561e-02 1.5633e+00 1.1369e-13
   240 6.9012e-04 4.0063e+01    1.7196e-02 1.6078e+00 4.2633e-14
   192 3.4559e-04 4.0961e+01    8.5552e-03 1.6546e+00 2.8422e-14
   144 1.4762e-04 4.0455e+01    3.1936e-03 1.8700e+00 2.8422e-14
    96 4.9653e-05 3.5637e+01    8.9450e-04 1.9782e+00 1.4211e-14
    48 9.7910e-06 2.2591e+01    8.4844e-05 2.6069e+00 7.1054e-15
];

% Maximum difference between reference and your implementation: 3.552714e-13.
