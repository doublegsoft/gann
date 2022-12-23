/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#ifndef __GANN_H__
#define __GANN_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef NULL
#define NULL              0
#endif

#define GANN_W2V_DEBUG

typedef unsigned int              uint;
typedef unsigned long             ulong;
typedef unsigned long long        ullong;
typedef long long                 llong;
typedef float                     real;


double
gnn_num_random(double mu, double sigma);

void
gnn_vec_print(double const* vec, uint size);

double*
gnn_vec_new(uint size, double seed);

void
gnn_vec_copy(double* dst, const double* src, uint size);

void
gnn_vec_add(double* dst, const double* addend, uint size);

void
gnn_vec_subtract(double* dst, const double* subtrahend, uint size);

void
gnn_vec_multiply(double* dst, const double* multiplicand, uint size);

void
gnn_vec_divide(double* dst, const double* dividend, uint size);

void
gnn_vec_add_scalar(double* dst, double addend, uint size);

void
gnn_vec_subtract_scalar(double* dst, double subtrahend, uint size);

void
gnn_vec_multiply_scalar(double* dst, double multiplicand, uint size);

void
gnn_vec_divide_scalar(double* dst, double dividend, uint size);


#ifdef __cplusplus
}
#endif

#endif
