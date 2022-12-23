/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#include <stdio.h>
#include <math.h>
#include <limits.h>

#include "gann.h"

/*!
** Gaussian generator:
**   https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
*/
double
gnn_num_random(double mu, double sigma)
{

  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
  {
    call = !call;
    return (mu + sigma * (double) X2);
  }

  do
  {
    U1 = -1 + ((double) rand () / RAND_MAX) * 2;
    U2 = -1 + ((double) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  } while ( W >= 1 || W == 0 );

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
}

void
gnn_vec_print(double const* vec, uint size)
{
  int i = 0;
  printf("[");
  for (i = 0; i < size; i++)
  {
    if (i > 0) printf(",");
    printf("%f", vec[i]);
  }
  printf("]\n");
}

double*
gnn_vec_new(uint size, double random)
{
  int         l       = 0;
  double*     ret;
  ret = (double*) calloc(size, sizeof(double));

  while ( l < size )
  {
    if (random <= 0)
      ret[l] = 0;
    else
      ret[l] = gnn_num_random(0,1) / sqrt(random / 5);
    ++l;
  }

  return ret;
}

void
gnn_vec_copy(double* dst, const double* src, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] = src[i];
}

void
gnn_vec_add(double* dst, const double* addend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] += addend[i];
}

void
gnn_vec_subtract(double* dst, const double* subtrahend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] -= subtrahend[i];
}

void
gnn_vec_multiply(double* dst, const double* multiplicand, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] *= multiplicand[i];
}

void
gnn_vec_divide(double* dst, const double* dividend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] /= dividend[i];
}

void
gnn_vec_add_scalar(double* dst, double addend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] += addend;
}

void
gnn_vec_subtract_scalar(double* dst, double subtrahend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] -= subtrahend;
}

void
gnn_vec_multiply_scalar(double* dst, double multiplicand, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] *= multiplicand;
}

void
gnn_vec_divide_scalar(double* dst, double dividend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] /= dividend;
}
