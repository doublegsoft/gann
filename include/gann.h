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


float
gnn_num_random(float mu, float sigma);

void
gnn_vec_print(float const* vec, uint size);

float*
gnn_vec_new(uint size, float seed);

void
gnn_vec_copy(float* dst, const float* src, uint size);

void
gnn_vec_add(float* dst, const float* addend, uint size);

void
gnn_vec_subtract(float* dst, const float* subtrahend, uint size);

void
gnn_vec_multiply(float* dst, const float* multiplicand, uint size);

void
gnn_vec_divide(float* dst, const float* dividend, uint size);

void
gnn_vec_add_scalar(float* dst, float addend, uint size);

void
gnn_vec_subtract_scalar(float* dst, float subtrahend, uint size);

void
gnn_vec_multiply_scalar(float* dst, float multiplicand, uint size);

void
gnn_vec_divide_scalar(float* dst, float dividend, uint size);


#ifdef __cplusplus
}
#endif

#endif
