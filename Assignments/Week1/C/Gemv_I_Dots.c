#define alpha( i,j ) A[ (j)*ldA + i ]   // map alpha( i,j ) to array A 
#define chi( i )  x[ (i)*incx ]         // map chi( i )  to array x
#define psi( i )  y[ (i)*incy ]         // map psi( i )  to array y

void Dots( int, const double *, int, const double *, int, double * );
// size of vectors to dot product,
// location of first vector
// increment of first vector
// location of 2nd vector
// increment of 2nd vector
// location for answer

// y = Ax+y
void MyGemv( int m, int n, double *A, int ldA,
           double *x, int incx, double *y, int incy )
{
  for ( int i=0; i<m; i++ )
    Dots( n,&(alpha(i,0)), ldA, x, incx, &(psi(i)));
}