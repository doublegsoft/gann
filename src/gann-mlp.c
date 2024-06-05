/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "gann-mlp.h"

#define LOOKUP_SIZE 4096

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif

static const float sigmoid_dom_min = -15.0;
static const float sigmoid_dom_max = 15.0;
static float interval;
static float lookup[LOOKUP_SIZE];

static float
gnn_mlp_sigmoid(float a)
{
  if (a < -45.0) return 0;
  if (a > 45.0) return 1;
  return 1.0 / (1 + exp(-a));
}

static float
gnn_mlp_sigmoid_lookup(float a) {
  assert(!isnan(a));

  if (a < sigmoid_dom_min) return lookup[0];
  if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

  size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

  /* Because floating point... */
  if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

  return lookup[j];
}

static void
gnn_mlp_sigmoid_init(void) {
  const float f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
  int i;

  interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
  for (i = 0; i < LOOKUP_SIZE; ++i) {
    lookup[i] = gnn_mlp_sigmoid(sigmoid_dom_min + f * i);
  }
}

static float
gnn_mlp_linear(float a)
{
  return a;
}

static float
gnn_mlp_threshold(float a)
{
  return a > 0;
}

static void
gnn_mlp_randomize(gnn_mlp_t* mlp) {
  int i;
  for (i = 0; i < mlp->total_weights; ++i) {
    float r = GNN_MLP_RANDOM();
    /* Sets weights from -0.5 to 0.5. */
    mlp->weights[i] = r - 0.5;
  }
}

gnn_mlp_t*
gnn_mlp_new(int input_number,
            int hidden_layer_number,
            int hidden_neuron_number,
            int output_number)
{
  if (hidden_layer_number < 0) return NULL;
  if (input_number < 1) return NULL;
  if (output_number < 1) return NULL;
  if (hidden_layer_number > 0 && hidden_neuron_number < 1) return NULL;


  const int hidden_weights = hidden_layer_number ?
                             (input_number + 1) * hidden_neuron_number + (hidden_layer_number - 1) * (hidden_neuron_number + 1) * hidden_neuron_number :
                             0;
  const int output_weights = (hidden_layer_number ? (hidden_neuron_number + 1) : (input_number + 1)) * output_number;
  const int total_weights = (hidden_weights + output_weights);

  const int total_neurons = (input_number + hidden_neuron_number * hidden_layer_number + output_number);

  /*!
  ** allocate extra size for weights, outputs, and biases.
  */
  const int size = sizeof(gnn_mlp_t) + sizeof(float) * (total_weights + total_neurons + (total_neurons - input_number));

  gnn_mlp_t*  ret = (gnn_mlp_t*)malloc(size);
  if (!ret) return NULL;

  ret->input_number = input_number;
  ret->hidden_layer_number = hidden_layer_number;
  ret->hidden_neuron_number = hidden_neuron_number;
  ret->output_number = output_number;

  ret->total_weights = total_weights;
  ret->total_neurons = total_neurons;

  ret->weights = (float*)((char*)ret + sizeof(gnn_mlp_t));
  ret->outputs = ret->weights + ret->total_weights;
  ret->biases = ret->outputs + ret->total_neurons;

  gnn_mlp_randomize(ret);

  ret->activation_hidden = gnn_mlp_sigmoid_lookup;
  ret->activation_output = gnn_mlp_sigmoid_lookup;

  gnn_mlp_sigmoid_init();

  return ret;
}

void
gnn_mlp_free(gnn_mlp_t* mlp)
{
  free(mlp);
}

float const*
gnn_mlp_forward(gnn_mlp_t const* mlp,
                float const* inputs)
{
  float const* w = mlp->weights;
  float* o = mlp->outputs + mlp->input_number;
  float const* i = mlp->outputs;

  /*!
  ** copy the inputs to the scratch area, where we also store each neuron's
  ** output, for consistency. This way the first layer isn't a special case.
  */
  memcpy(mlp->outputs, inputs, sizeof(float) * mlp->input_number);

  int h, j, k;

  if (!mlp->hidden_layer_number) {
    float *ret = o;
    for (j = 0; j < mlp->output_number; ++j) {
      float sum = *w++ * -1.0;
      for (k = 0; k < mlp->input_number; ++k) {
        sum += *w++ * i[k];
      }
      *o++ = mlp->activation_output(sum);
    }
    return ret;
  }

  /*!
  ** inputs -> hidden neurons in first layer
  */
  for (j = 0; j < mlp->hidden_neuron_number; ++j) {
    float sum = *w++ * -1.0;
    for (k = 0; k < mlp->input_number; ++k) {
      sum += *w++ * i[k];
    }
    *o++ = mlp->activation_hidden(sum);
  }

  i += mlp->input_number;

  /*!
  ** hidden layers
  */
  for (h = 1; h < mlp->hidden_layer_number; ++h) {
    for (j = 0; j < mlp->hidden_neuron_number; ++j) {
      float sum = *w++ * -1.0;
      for (k = 0; k < mlp->hidden_neuron_number; ++k) {
        sum += *w++ * i[k];
      }
      *o++ = mlp->activation_hidden(sum);
    }

    i += mlp->hidden_neuron_number;
  }

  float const *ret = o;

  /*!
  ** output layer
  */
  for (j = 0; j < mlp->output_number; ++j) {
    float sum = *w++ * -1.0;
    for (k = 0; k < mlp->hidden_neuron_number; ++k) {
      sum += *w++ * i[k];
    }
    *o++ = mlp->activation_output(sum);
  }

  /* Sanity check that we used all weights and wrote all outputs. */
  assert(w - mlp->weights == mlp->total_weights);
  assert(o - mlp->outputs == mlp->total_neurons);

  return ret;
}

void
gnn_mlp_train(gnn_mlp_t   const*        mlp,
              float       const*        inputs,
              float       const*        desired_outputs,
              float                     learning_rate)
{
  /*!
  ** at the beginning, we must run the network forward.
  */
  gnn_mlp_forward(mlp, inputs);

  int h, j, k;

  /* First set the output layer deltas. */
  {
    float const *o = mlp->outputs + mlp->input_number
        + mlp->hidden_neuron_number * mlp->hidden_layer_number; /* first output. */
    float *d = mlp->biases + mlp->hidden_neuron_number * mlp->hidden_layer_number; /* first bias. */
    float const *t = desired_outputs; /* first desired output. */

    /* set output layer deltas. */
    if (mlp->activation_output == gnn_mlp_linear) {
      for (j = 0; j < mlp->output_number; ++j) {
        *d++ = *t++ - *o++;
      }
    } else {
      for (j = 0; j < mlp->output_number; ++j) {
        *d++ = (*t - *o) * *o * (1.0 - *o);
        ++o;
        ++t;
      }
    }
  }

  /*!
  ** Set hidden layer deltas, start on last layer and work backwards.
  ** Note that loop is skipped in the case of hidden_layers == 0.
  */
  for (h = mlp->hidden_layer_number - 1; h >= 0; --h) {

    /* Find first output and delta in this layer. */
    float const *o = mlp->outputs + mlp->input_number + (h * mlp->hidden_neuron_number);
    float *d = mlp->biases + (h * mlp->hidden_neuron_number);

    /* Find first delta in following layer (which may be hidden or output). */
    float const* const dd = mlp->biases + ((h + 1) * mlp->hidden_neuron_number);

    /* Find first weight in following layer (which may be hidden or output). */
    float const* const ww = mlp->weights + ((mlp->input_number + 1) * mlp->hidden_neuron_number)
        + ((mlp->hidden_neuron_number + 1) * mlp->hidden_neuron_number * (h));

    for (j = 0; j < mlp->hidden_neuron_number; ++j) {

      float delta = 0;

      for (k = 0;
           k < (h == mlp->hidden_layer_number - 1 ? mlp->output_number : mlp->hidden_neuron_number); ++k) {
        const float forward_delta = dd[k];
        const int windex = k * (mlp->hidden_neuron_number + 1) + (j + 1);
        const float forward_weight = ww[windex];
        delta += forward_delta * forward_weight;
      }

      *d = *o * (1.0 - *o) * delta;
      ++d;
      ++o;
    }
  }

  /* Train the outputs. */
  {
    /* Find first output delta. */
    float const *d = mlp->biases + mlp->hidden_neuron_number * mlp->hidden_layer_number; /* First output delta. */

    /* Find first weight to first output delta. */
    float *w = mlp->weights
        + (mlp->hidden_layer_number ?
            ((mlp->input_number + 1) * mlp->hidden_neuron_number
                + (mlp->hidden_neuron_number + 1) * mlp->hidden_neuron_number * (mlp->hidden_layer_number - 1)) :
            (0));

    /* Find first output in previous layer. */
    float const *const i = mlp->outputs
        + (mlp->hidden_layer_number ?
            (mlp->input_number + (mlp->hidden_neuron_number) * (mlp->hidden_layer_number - 1)) : 0);

    /* Set output layer weights. */
    for (j = 0; j < mlp->output_number; ++j) {
      *w++ += *d * learning_rate * -1.0;
      for (k = 1; k < (mlp->hidden_layer_number ? mlp->hidden_neuron_number : mlp->input_number) + 1;
          ++k) {
        *w++ += *d * learning_rate * i[k - 1];
      }

      ++d;
    }

    assert(w - mlp->weights == mlp->total_weights);
  }

  /* Train the hidden layers. */
  for (h = mlp->hidden_layer_number - 1; h >= 0; --h) {

    /* Find first delta in this layer. */
    float const *d = mlp->biases + (h * mlp->hidden_neuron_number);

    /* Find first input to this layer. */
    float const *i = mlp->outputs
        + (h ? (mlp->input_number + mlp->hidden_neuron_number * (h - 1)) : 0);

    /* Find first weight to this layer. */
    float *w = mlp->weights
        + (h ?
            ((mlp->input_number + 1) * mlp->hidden_neuron_number
                + (mlp->hidden_neuron_number + 1) * (mlp->hidden_neuron_number) * (h - 1)) :
            0);

    for (j = 0; j < mlp->hidden_neuron_number; ++j) {
      *w++ += *d * learning_rate * -1.0;
      for (k = 1; k < (h == 0 ? mlp->input_number : mlp->hidden_neuron_number) + 1; ++k) {
        *w++ += *d * learning_rate * i[k - 1];
      }
      ++d;
    }
  }
}

gnn_mlp_t*
gnn_mlp_read(FILE* in)
{
  int inputs, hidden_layers, hidden, outputs;
  int rc;

  errno = 0;
  rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
  if (rc < 4 || errno != 0) {
      perror("fscanf");
      return NULL;
  }

  gnn_mlp_t* ret = gnn_mlp_new(inputs, hidden_layers, hidden, outputs);

  int i;
  for (i = 0; i < ret->total_weights; ++i)
  {
    errno = 0;
    rc = fscanf(in, " %e", ret->weights + i);
    if (rc < 1 || errno != 0)
    {
      perror("fscanf");
      gnn_mlp_free(ret);
      return NULL;
    }
  }

  return ret;
}

void
gnn_mlp_write(gnn_mlp_t const* mlp, FILE* out)
{
  fprintf(out, "%d %d %d %d", mlp->input_number, mlp->hidden_layer_number, mlp->hidden_neuron_number, mlp->output_number);
  int i;
  for (i = 0; i < mlp->total_weights; ++i)
    fprintf(out, " %.20e", mlp->weights[i]);

}

