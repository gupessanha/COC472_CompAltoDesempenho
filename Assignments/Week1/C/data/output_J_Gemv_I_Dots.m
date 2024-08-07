% number of repeats:% 3
% enter first, last, inc:% 48 480 48 
data = [
%  n          reference      |         current implementation 
%        time       GFLOPS   |    time       GFLOPS     diff 
   480 5.0768e-03 4.3567e+01    1.3488e-01 1.6399e+00 3.5527e-13
   432 3.3117e-03 4.8688e+01    9.5985e-02 1.6799e+00 3.1264e-13
   384 2.8775e-03 3.9355e+01    6.9632e-02 1.6264e+00 1.9895e-13
   336 1.7746e-03 4.2750e+01    4.5458e-02 1.6689e+00 1.7053e-13
   288 9.9288e-04 4.8119e+01    3.0168e-02 1.5837e+00 1.1369e-13
   240 6.3037e-04 4.3860e+01    1.5964e-02 1.7319e+00 4.2633e-14
   192 3.5834e-04 3.9503e+01    8.1283e-03 1.7415e+00 2.8422e-14
   144 1.4399e-04 4.1474e+01    2.8106e-03 2.1248e+00 2.8422e-14
    96 4.6342e-05 3.8183e+01    9.2253e-04 1.9181e+00 1.4211e-14
    48 9.3510e-06 2.3654e+01    9.0054e-05 2.4561e+00 7.1054e-15
];

% Maximum difference between reference and your implementation: 3.552714e-13.
