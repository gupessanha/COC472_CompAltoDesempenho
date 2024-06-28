#define alpha( i,j ) A[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#include<immintrin.h>

void Gemm_MRxNRKernel( int k, double *A, int ldA, double *B, int ldB,
		double *C, int ldC )
{
  /* Declare vector registers to hold 24x8 C and load them */
  __m256d gamma_0123_0 = _mm256_loadu_pd( &gamma( 0,0 ) );
  __m256d gamma_0123_1 = _mm256_loadu_pd( &gamma( 0,1 ) );
  __m256d gamma_0123_2 = _mm256_loadu_pd( &gamma( 0,2 ) );
  __m256d gamma_0123_3 = _mm256_loadu_pd( &gamma( 0,3 ) );
  __m256d gamma_0123_4 = _mm256_loadu_pd( &gamma( 0,4 ) );
  __m256d gamma_0123_5 = _mm256_loadu_pd( &gamma( 0,5 ) );
  __m256d gamma_0123_6 = _mm256_loadu_pd( &gamma( 0,6 ) );
  __m256d gamma_0123_7 = _mm256_loadu_pd( &gamma( 0,7 ) );

  __m256d gamma_4567_0 = _mm256_loadu_pd( &gamma( 4,0 ) );
  __m256d gamma_4567_1 = _mm256_loadu_pd( &gamma( 4,1 ) );
  __m256d gamma_4567_2 = _mm256_loadu_pd( &gamma( 4,2 ) );
  __m256d gamma_4567_3 = _mm256_loadu_pd( &gamma( 4,3 ) );
  __m256d gamma_4567_4 = _mm256_loadu_pd( &gamma( 4,4 ) );
  __m256d gamma_4567_5 = _mm256_loadu_pd( &gamma( 4,5 ) );
  __m256d gamma_4567_6 = _mm256_loadu_pd( &gamma( 4,6 ) );
  __m256d gamma_4567_7 = _mm256_loadu_pd( &gamma( 4,7 ) );

  __m256d gamma_891011_0 = _mm256_loadu_pd( &gamma( 8,0 ) );
  __m256d gamma_891011_1 = _mm256_loadu_pd( &gamma( 8,1 ) );
  __m256d gamma_891011_2 = _mm256_loadu_pd( &gamma( 8,2 ) );
  __m256d gamma_891011_3 = _mm256_loadu_pd( &gamma( 8,3 ) );
  __m256d gamma_891011_4 = _mm256_loadu_pd( &gamma( 8,4 ) );
  __m256d gamma_891011_5 = _mm256_loadu_pd( &gamma( 8,5 ) );
  __m256d gamma_891011_6 = _mm256_loadu_pd( &gamma( 8,6 ) );
  __m256d gamma_891011_7 = _mm256_loadu_pd( &gamma( 8,7 ) );

  __m256d gamma_12131415_0 = _mm256_loadu_pd( &gamma( 12,0 ) );
  __m256d gamma_12131415_1 = _mm256_loadu_pd( &gamma( 12,1 ) );
  __m256d gamma_12131415_2 = _mm256_loadu_pd( &gamma( 12,2 ) );
  __m256d gamma_12131415_3 = _mm256_loadu_pd( &gamma( 12,3 ) );
  __m256d gamma_12131415_4 = _mm256_loadu_pd( &gamma( 12,4 ) );
  __m256d gamma_12131415_5 = _mm256_loadu_pd( &gamma( 12,5 ) );
  __m256d gamma_12131415_6 = _mm256_loadu_pd( &gamma( 12,6 ) );
  __m256d gamma_12131415_7 = _mm256_loadu_pd( &gamma( 12,7 ) );

  __m256d gamma_16171819_0 = _mm256_loadu_pd( &gamma( 16,0 ) );
  __m256d gamma_16171819_1 = _mm256_loadu_pd( &gamma( 16,1 ) );
  __m256d gamma_16171819_2 = _mm256_loadu_pd( &gamma( 16,2 ) );
  __m256d gamma_16171819_3 = _mm256_loadu_pd( &gamma( 16,3 ) );
  __m256d gamma_16171819_4 = _mm256_loadu_pd( &gamma( 16,4 ) );
  __m256d gamma_16171819_5 = _mm256_loadu_pd( &gamma( 16,5 ) );
  __m256d gamma_16171819_6 = _mm256_loadu_pd( &gamma( 16,6 ) );
  __m256d gamma_16171819_7 = _mm256_loadu_pd( &gamma( 16,7 ) );

  __m256d gamma_20212223_0 = _mm256_loadu_pd( &gamma( 20,0 ) );
  __m256d gamma_20212223_1 = _mm256_loadu_pd( &gamma( 20,1 ) );
  __m256d gamma_20212223_2 = _mm256_loadu_pd( &gamma( 20,2 ) );
  __m256d gamma_20212223_3 = _mm256_loadu_pd( &gamma( 20,3 ) );
  __m256d gamma_20212223_4 = _mm256_loadu_pd( &gamma( 20,4 ) );
  __m256d gamma_20212223_5 = _mm256_loadu_pd( &gamma( 20,5 ) );
  __m256d gamma_20212223_6 = _mm256_loadu_pd( &gamma( 20,6 ) );
  __m256d gamma_20212223_7 = _mm256_loadu_pd( &gamma( 20,7 ) );

  for ( int p=0; p<k; p++ ){
    /* Declare vector register for load/broadcasting beta( p,j ) */
    __m256d beta_p_j;

    /* Declare vector registers to hold the current column of A and load
       them with the twenty-four elements of that column. */
    __m256d alpha_0123_p = _mm256_loadu_pd( &alpha( 0,p ) );
    __m256d alpha_4567_p = _mm256_loadu_pd( &alpha( 4,p ) );
    __m256d alpha_891011_p = _mm256_loadu_pd( &alpha( 8,p ) );
    __m256d alpha_12131415_p = _mm256_loadu_pd( &alpha( 12,p ) );
    __m256d alpha_16171819_p = _mm256_loadu_pd( &alpha( 16,p ) );
    __m256d alpha_20212223_p = _mm256_loadu_pd( &alpha( 20,p ) );

    /* Repeat the updates for each column of C (0 to 7) */
    for (int j = 0; j < 8; j++) {
      beta_p_j = _mm256_broadcast_sd( &beta( p, j ) );

      gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
      gamma_4567_0 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
      gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );
      gamma_12131415_0 = _mm256_fmadd_pd( alpha_12131415_p, beta_p_j, gamma_12131415_0 );
      gamma_16171819_0 = _mm256_fmadd_pd( alpha_16171819_p, beta_p_j, gamma_16171819_0 );
      gamma_20212223_0 = _mm256_fmadd_pd( alpha_20212223_p, beta_p_j, gamma_20212223_0 );
    }
  }

  /* Store the updated results */
  _mm256_storeu_pd( &gamma( 0,0 ), gamma_0123_0 );
  _mm256_storeu_pd( &gamma( 0,1 ), gamma_0123_1 );
  _mm256_storeu_pd( &gamma( 0,2 ), gamma_0123_2 );
  _mm256_storeu_pd( &gamma( 0,3 ), gamma_0123_3 );
  _mm256_storeu_pd( &gamma( 0,4 ), gamma_0123_4 );
  _mm256_storeu_pd( &gamma( 0,5 ), gamma_0123_5 );
  _mm256_storeu_pd( &gamma( 0,6 ), gamma_0123_6 );
  _mm256_storeu_pd( &gamma( 0,7 ), gamma_0123_7 );

  _mm256_storeu_pd( &gamma( 4,0 ), gamma_4567_0 );
  _mm256_storeu_pd( &gamma( 4,1 ), gamma_4567_1 );
  _mm256_storeu_pd( &gamma( 4,2 ), gamma_4567_2 );
  _mm256_storeu_pd( &gamma( 4,3 ), gamma_4567_3 );
  _mm256_storeu_pd( &gamma( 4,4 ), gamma_4567_4 );
  _mm256_storeu_pd( &gamma( 4,5 ), gamma_4567_5 );
  _mm256_storeu_pd( &gamma( 4,6 ), gamma_4567_6 );
  _mm256_storeu_pd( &gamma( 4,7 ), gamma_4567_7 );

  _mm256_storeu_pd( &gamma( 8,0 ), gamma_891011_0 );
  _mm256_storeu_pd( &gamma( 8,1 ), gamma_891011_1 );
  _mm256_storeu_pd( &gamma( 8,2 ), gamma_891011_2 );
  _mm256_storeu_pd( &gamma( 8,3 ), gamma_891011_3 );
  _mm256_storeu_pd( &gamma( 8,4 ), gamma_891011_4 );
  _mm256_storeu_pd( &gamma( 8,5 ), gamma_891011_5 );
  _mm256_storeu_pd( &gamma( 8,6 ), gamma_891011_6 );
  _mm256_storeu_pd( &gamma( 8,7 ), gamma_891011_7 );

  _mm256_storeu_pd( &gamma( 12,0 ), gamma_12131415_0 );
  _mm256_storeu_pd( &gamma( 12,1 ), gamma_12131415_1 );
  _mm256_storeu_pd( &gamma( 12,2 ), gamma_12131415_2 );
  _mm256_storeu_pd( &gamma( 12,3 ), gamma_12131415_3 );
  _mm256_storeu_pd( &gamma( 12,4 ), gamma_12131415_4 );
  _mm256_storeu_pd( &gamma( 12,5 ), gamma_12131415_5 );
  _mm256_storeu_pd( &gamma( 12,6 ), gamma_12131415_6 );
  _mm256_storeu_pd( &gamma( 12,7 ), gamma_12131415_7 );

  _mm256_storeu_pd( &gamma( 16,0 ), gamma_16171819_0 );
  _mm256_storeu_pd( &gamma( 16,1 ), gamma_16171819_1 );
  _mm256_storeu_pd( &gamma( 16,2 ), gamma_16171819_2 );
  _mm256_storeu_pd( &gamma( 16,3 ), gamma_16171819_3 );
  _mm256_storeu_pd( &gamma( 16,4 ), gamma_16171819_4 );
  _mm256_storeu_pd( &gamma( 16,5 ), gamma_16171819_5 );
  _mm256_storeu_pd( &gamma( 16,6 ), gamma_16171819_6 );
  _mm256_storeu_pd( &gamma( 16,7 ), gamma_16171819_7 );

  _mm256_storeu_pd( &gamma( 20,0 ), gamma_20212223_0 );
  _mm256_storeu_pd( &gamma( 20,1 ), gamma_20212223_1 );
  _mm256_storeu_pd( &gamma( 20,2 ), gamma_20212223_2 );
  _mm256_storeu_pd( &gamma( 20,3 ), gamma_20212223_3 );
  _mm256_storeu_pd( &gamma( 20,4 ), gamma_20212223_4 );
  _mm256_storeu_pd( &gamma( 20,5 ), gamma_20212223_5 );
  _mm256_storeu_pd( &gamma( 20,6 ), gamma_20212223_6 );
  _mm256_storeu_pd( &gamma( 20,7 ), gamma_20212223_7 );
}