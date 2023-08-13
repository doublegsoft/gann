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
float
gnn_num_random(float mu, float sigma)
{

  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;

  if (call == 1)
  {
    call = !call;
    return (mu + sigma * (float) X2);
  }

  do
  {
    U1 = -1 + ((float) rand () / RAND_MAX) * 2;
    U2 = -1 + ((float) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  } while ( W >= 1 || W == 0 );

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (float) X1);
}

void
gnn_vec_print(float const* vec, uint size)
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

float*
gnn_vec_new(uint size, float random)
{
  int         l       = 0;
  float*     ret;
  ret = (float*) calloc(size, sizeof(float));

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
gnn_vec_copy(float* dst, const float* src, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] = src[i];
}

void
gnn_vec_add(float* dst, const float* addend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] += addend[i];
}

void
gnn_vec_subtract(float* dst, const float* subtrahend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] -= subtrahend[i];
}

void
gnn_vec_multiply(float* dst, const float* multiplicand, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] *= multiplicand[i];
}

void
gnn_vec_divide(float* dst, const float* dividend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] /= dividend[i];
}

void
gnn_vec_add_scalar(float* dst, float addend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] += addend;
}

void
gnn_vec_subtract_scalar(float* dst, float subtrahend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] -= subtrahend;
}

void
gnn_vec_multiply_scalar(float* dst, float multiplicand, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] *= multiplicand;
}

void
gnn_vec_divide_scalar(float* dst, float dividend, uint size)
{
  int i = 0;
  for (i = 0; i < size; i++)
    dst[i] /= dividend;
}
