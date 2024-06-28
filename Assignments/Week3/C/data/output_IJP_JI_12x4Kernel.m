% number of repeats:% 3
% enter first, last, inc:% 48 1488 48 
data = [
%  n          reference      |         current implementation 
%        time       GFLOPS   |    time       GFLOPS     diff 
  1488 1.7224e-01 3.8257e+01    2.5600e-01 2.5740e+01 2.1600e-12
  1440 1.4599e-01 4.0908e+01    2.6031e-01 2.2941e+01 2.1032e-12
  1392 1.3027e-01 4.1409e+01    2.1272e-01 2.5359e+01 1.9327e-12
  1344 1.1602e-01 4.1852e+01    1.8725e-01 2.5930e+01 1.7621e-12
  1296 1.0754e-01 4.0481e+01    1.6471e-01 2.6432e+01 1.8190e-12
  1248 9.3103e-02 4.1755e+01    1.4915e-01 2.6065e+01 1.6485e-12
  1200 8.2691e-02 4.1794e+01    1.3181e-01 2.6220e+01 1.5348e-12
  1152 7.2205e-02 4.2347e+01    1.2329e-01 2.4801e+01 1.3642e-12
  1104 6.5129e-02 4.1320e+01    1.0380e-01 2.5926e+01 1.2506e-12
  1056 5.5958e-02 4.2088e+01    1.0798e-01 2.1812e+01 1.1937e-12
  1008 4.7090e-02 4.3499e+01    7.5146e-02 2.7259e+01 1.0232e-12
   960 4.1919e-02 4.2212e+01    6.9815e-02 2.5345e+01 1.0232e-12
   912 3.4554e-02 4.3906e+01    5.6109e-02 2.7039e+01 1.0800e-12
   864 2.9713e-02 4.3414e+01    4.8590e-02 2.6547e+01 8.5265e-13
   816 2.4364e-02 4.4602e+01    4.0179e-02 2.7046e+01 8.2423e-13
   768 2.0796e-02 4.3565e+01    3.8989e-02 2.3237e+01 7.1054e-13
   720 1.7328e-02 4.3080e+01    2.7080e-02 2.7567e+01 6.8212e-13
   672 1.4473e-02 4.1935e+01    2.2003e-02 2.7584e+01 5.6843e-13
   624 1.0558e-02 4.6028e+01    1.8028e-02 2.6954e+01 5.4001e-13
   576 8.4777e-03 4.5084e+01    1.4879e-02 2.5687e+01 4.8317e-13
   528 6.2865e-03 4.6830e+01    1.0814e-02 2.7224e+01 4.2633e-13
   480 4.6978e-03 4.7083e+01    7.4128e-03 2.9838e+01 3.1264e-13
   432 3.3488e-03 4.8150e+01    4.9560e-03 3.2535e+01 2.7001e-13
   384 2.3996e-03 4.7195e+01    3.8042e-03 2.9769e+01 2.2737e-13
   336 1.5510e-03 4.8914e+01    2.5342e-03 2.9937e+01 1.7053e-13
   288 9.7827e-04 4.8837e+01    1.4462e-03 3.3035e+01 9.9476e-14
   240 9.1480e-04 3.0223e+01    8.3408e-04 3.3148e+01 4.2633e-14
   192 2.9585e-04 4.7848e+01    3.7832e-04 3.7418e+01 2.8422e-14
   144 1.3052e-04 4.5755e+01    1.2902e-04 4.6286e+01 2.8422e-14
    96 4.3554e-05 4.0627e+01    3.8724e-05 4.5694e+01 1.0658e-14
    48 8.6320e-06 2.5624e+01    4.5700e-06 4.8399e+01 5.3291e-15
];

% Maximum difference between reference and your implementation: 2.160050e-12.