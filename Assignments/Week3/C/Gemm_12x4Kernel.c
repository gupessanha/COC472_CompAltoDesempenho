#define alpha( i,j ) A[ (j)*ldA + (i) ]   // mapear alpha( i,j ) para a matriz A
#define beta( i,j )  B[ (j)*ldB + (i) ]   // mapear beta( i,j ) para a matriz B
#define gamma( i,j ) C[ (j)*ldC + (i) ]   // mapear gamma( i,j ) para a matriz C

#include<immintrin.h>

void Gemm_MRxNRKernel( int k, double *A, int ldA, double *B, int ldB,
		double *C, int ldC )
{
  /* Declarar registradores vetoriais para manter o bloco 12x4 de C e carregá-los */
  __m256d gamma_0123_0 = _mm256_loadu_pd( &gamma( 0,0 ) );
  __m256d gamma_0123_1 = _mm256_loadu_pd( &gamma( 0,1 ) );
  __m256d gamma_0123_2 = _mm256_loadu_pd( &gamma( 0,2 ) );
  __m256d gamma_0123_3 = _mm256_loadu_pd( &gamma( 0,3 ) );

  __m256d gamma_4567_0 = _mm256_loadu_pd( &gamma( 4,0 ) );
  __m256d gamma_4567_1 = _mm256_loadu_pd( &gamma( 4,1 ) );
  __m256d gamma_4567_2 = _mm256_loadu_pd( &gamma( 4,2 ) );
  __m256d gamma_4567_3 = _mm256_loadu_pd( &gamma( 4,3 ) );

  __m256d gamma_891011_0 = _mm256_loadu_pd( &gamma( 8,0 ) );
  __m256d gamma_891011_1 = _mm256_loadu_pd( &gamma( 8,1 ) );
  __m256d gamma_891011_2 = _mm256_loadu_pd( &gamma( 8,2 ) );
  __m256d gamma_891011_3 = _mm256_loadu_pd( &gamma( 8,3 ) );

  for ( int p=0; p<k; p++ ){
    /* Declarar registradores vetoriais para carregar/difundir beta( p,j ) */
    __m256d beta_p_j;

    /* Declarar registradores vetoriais para manter a coluna atual de A e carregar
       eles com os doze elementos dessa coluna. */
    __m256d alpha_0123_p = _mm256_loadu_pd( &alpha( 0,p ) );
    __m256d alpha_4567_p = _mm256_loadu_pd( &alpha( 4,p ) );
    __m256d alpha_891011_p = _mm256_loadu_pd( &alpha( 8,p ) );

    /* Carregar/difundir beta( p,0 ). */
    beta_p_j = _mm256_broadcast_sd( &beta( p, 0) );

    /* Atualizar a primeira coluna de C com a coluna atual de A vezes
       beta( p,0 ) */
    gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
    gamma_4567_0 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
    gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

    /* Repetir para a segunda, terceira e quarta colunas de C. Note que a
       coluna atual de A não precisa ser recarregada. */

    /* Carregar/difundir beta( p,1 ). */
    beta_p_j = _mm256_broadcast_sd( &beta( p, 1) );

    /* Atualizar a segunda coluna de C com a coluna atual de A vezes
       beta( p,1 ) */
    gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );
    gamma_4567_1 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_1 );
    gamma_891011_1 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_1 );

    /* Carregar/difundir beta( p,2 ). */
    beta_p_j = _mm256_broadcast_sd( &beta( p, 2) );

    /* Atualizar a terceira coluna de C com a coluna atual de A vezes
       beta( p,2 ) */
    gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );
    gamma_4567_2 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_2 );
    gamma_891011_2 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_2 );

    /* Carregar/difundir beta( p,3 ). */
    beta_p_j = _mm256_broadcast_sd( &beta( p, 3) );

    /* Atualizar a quarta coluna de C com a coluna atual de A vezes
       beta( p,3 ) */
    gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );
    gamma_4567_3 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_3 );
    gamma_891011_3 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_3 );
  }

  /* Armazenar os resultados atualizados */
  _mm256_storeu_pd( &gamma(0,0), gamma_0123_0 );
  _mm256_storeu_pd( &gamma(0,1), gamma_0123_1 );
  _mm256_storeu_pd( &gamma(0,2), gamma_0123_2 );
  _mm256_storeu_pd( &gamma(0,3), gamma_0123_3 );

  _mm256_storeu_pd( &gamma(4,0), gamma_4567_0 );
  _mm256_storeu_pd( &gamma(4,1), gamma_4567_1 );
  _mm256_storeu_pd( &gamma(4,2), gamma_4567_2 );
  _mm256_storeu_pd( &gamma(4,3), gamma_4567_3 );

  _mm256_storeu_pd( &gamma(8,0), gamma_891011_0 );
  _mm256_storeu_pd( &gamma(8,1), gamma_891011_1 );
  _mm256_storeu_pd( &gamma(8,2), gamma_891011_2 );
  _mm256_storeu_pd( &gamma(8,3), gamma_891011_3 );
}
