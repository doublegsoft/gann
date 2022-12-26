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

static float
gnn_mlp_sigmoid(float a)
{
  if (a < -45.0) return 0;
  if (a > 45.0) return 1;
  return 1.0 / (1 + exp(-a));
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
    mlp->weight[i] = r - 0.5;
  }
}

gnn_mlp_t*
gnn_mlp_new(int inputs, int hidden_layers, int hidden, int outputs)
{
  if (hidden_layers < 0) return 0;
  if (inputs < 1) return 0;
  if (outputs < 1) return 0;
  if (hidden_layers > 0 && hidden < 1) return 0;


  const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
  const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
  const int total_weights = (hidden_weights + output_weights);

  const int total_neurons = (inputs + hidden * hidden_layers + outputs);

  /* Allocate extra size for weights, outputs, and deltas. */
  const int size = sizeof(gnn_mlp_t) + sizeof(float) * (total_weights + total_neurons + (total_neurons - inputs));
  gnn_mlp_t*  ret = malloc(size);
  if (!ret) return 0;

  ret->inputs = inputs;
  ret->hidden_layers = hidden_layers;
  ret->hidden = hidden;
  ret->outputs = outputs;

  ret->total_weights = total_weights;
  ret->total_neurons = total_neurons;

  /* set pointers. */
  ret->weight = (float*)((char*)ret + sizeof(gnn_mlp_t));
  ret->output = ret->weight + ret->total_weights;
  ret->delta = ret->output + ret->total_neurons;

  gnn_mlp_randomize(ret);

  ret->activation_hidden = gnn_mlp_sigmoid;
  ret->activation_output = gnn_mlp_sigmoid;

//  gnn_mlp_init_sigmoid_lookup(ret);

  return ret;
}

void
gnn_mlp_free(gnn_mlp_t* mlp)
{
  free(mlp);
}

float const*
gnn_mlp_run(gnn_mlp_t const* mlp,
            float const* inputs)
{
  float const* w = mlp->weight;
  float* o = mlp->output + mlp->inputs;
  float const* i = mlp->output;

  /*!
  ** copy the inputs to the scratch area, where we also store each neuron's
  ** output, for consistency. This way the first layer isn't a special case.
  */
  memcpy(mlp->output, inputs, sizeof(float) * mlp->inputs);

  int h, j, k;

  if (!mlp->hidden_layers) {
    float *ret = o;
    for (j = 0; j < mlp->outputs; ++j) {
      float sum = *w++ * -1.0;
      for (k = 0; k < mlp->inputs; ++k) {
        sum += *w++ * i[k];
      }
      *o++ = mlp->activation_output(sum);
    }

    return ret;
  }

  /* Figure input layer */
  for (j = 0; j < mlp->hidden; ++j) {
    float sum = *w++ * -1.0;
    for (k = 0; k < mlp->inputs; ++k) {
      sum += *w++ * i[k];
    }
    *o++ = mlp->activation_hidden(sum);
  }

  i += mlp->inputs;

  /* Figure hidden layers, if any. */
  for (h = 1; h < mlp->hidden_layers; ++h) {
    for (j = 0; j < mlp->hidden; ++j) {
      float sum = *w++ * -1.0;
      for (k = 0; k < mlp->hidden; ++k) {
        sum += *w++ * i[k];
      }
      *o++ = mlp->activation_hidden(sum);
    }

    i += mlp->hidden;
  }

  float const *ret = o;

  /* Figure output layer. */
  for (j = 0; j < mlp->outputs; ++j) {
    float sum = *w++ * -1.0;
    for (k = 0; k < mlp->hidden; ++k) {
      sum += *w++ * i[k];
    }
    *o++ = mlp->activation_output(sum);
  }

  /* Sanity check that we used all weights and wrote all outputs. */
  assert(w - mlp->weight == mlp->total_weights);
  assert(o - mlp->output == mlp->total_neurons);

  return ret;
}

void
gnn_mlp_train(gnn_mlp_t   const*        mlp,
              float      const*        inputs,
              float      const*        desired_outputs,
              float                    learning_rate)
{
  /* To begin with, we must run the network forward. */
  gnn_mlp_run(mlp, inputs);

  int h, j, k;

  /* First set the output layer deltas. */
  {
    float const *o = mlp->output + mlp->inputs
        + mlp->hidden * mlp->hidden_layers; /* First output. */
    float *d = mlp->delta + mlp->hidden * mlp->hidden_layers; /* First delta. */
    float const *t = desired_outputs; /* First desired output. */

    /* Set output layer deltas. */
    if (mlp->activation_output == gnn_mlp_linear) {
      for (j = 0; j < mlp->outputs; ++j) {
        *d++ = *t++ - *o++;
      }
    } else {
      for (j = 0; j < mlp->outputs; ++j) {
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
  for (h = mlp->hidden_layers - 1; h >= 0; --h) {

    /* Find first output and delta in this layer. */
    float const *o = mlp->output + mlp->inputs + (h * mlp->hidden);
    float *d = mlp->delta + (h * mlp->hidden);

    /* Find first delta in following layer (which may be hidden or output). */
    float const* const dd = mlp->delta + ((h + 1) * mlp->hidden);

    /* Find first weight in following layer (which may be hidden or output). */
    float const* const ww = mlp->weight + ((mlp->inputs + 1) * mlp->hidden)
        + ((mlp->hidden + 1) * mlp->hidden * (h));

    for (j = 0; j < mlp->hidden; ++j) {

      float delta = 0;

      for (k = 0;
           k < (h == mlp->hidden_layers - 1 ? mlp->outputs : mlp->hidden); ++k) {
        const float forward_delta = dd[k];
        const int windex = k * (mlp->hidden + 1) + (j + 1);
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
    float const *d = mlp->delta + mlp->hidden * mlp->hidden_layers; /* First output delta. */

    /* Find first weight to first output delta. */
    float *w = mlp->weight
        + (mlp->hidden_layers ?
            ((mlp->inputs + 1) * mlp->hidden
                + (mlp->hidden + 1) * mlp->hidden * (mlp->hidden_layers - 1)) :
            (0));

    /* Find first output in previous layer. */
    float const *const i = mlp->output
        + (mlp->hidden_layers ?
            (mlp->inputs + (mlp->hidden) * (mlp->hidden_layers - 1)) : 0);

    /* Set output layer weights. */
    for (j = 0; j < mlp->outputs; ++j) {
      *w++ += *d * learning_rate * -1.0;
      for (k = 1; k < (mlp->hidden_layers ? mlp->hidden : mlp->inputs) + 1;
          ++k) {
        *w++ += *d * learning_rate * i[k - 1];
      }

      ++d;
    }

    assert(w - mlp->weight == mlp->total_weights);
  }

  /* Train the hidden layers. */
  for (h = mlp->hidden_layers - 1; h >= 0; --h) {

    /* Find first delta in this layer. */
    float const *d = mlp->delta + (h * mlp->hidden);

    /* Find first input to this layer. */
    float const *i = mlp->output
        + (h ? (mlp->inputs + mlp->hidden * (h - 1)) : 0);

    /* Find first weight to this layer. */
    float *w = mlp->weight
        + (h ?
            ((mlp->inputs + 1) * mlp->hidden
                + (mlp->hidden + 1) * (mlp->hidden) * (h - 1)) :
            0);

    for (j = 0; j < mlp->hidden; ++j) {
      *w++ += *d * learning_rate * -1.0;
      for (k = 1; k < (h == 0 ? mlp->inputs : mlp->hidden) + 1; ++k) {
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
    rc = fscanf(in, " %e", ret->weight + i);
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
  fprintf(out, "%d %d %d %d", mlp->inputs, mlp->hidden_layers, mlp->hidden, mlp->outputs);
  int i;
  for (i = 0; i < mlp->total_weights; ++i)
    fprintf(out, " %.20e", mlp->weight[i]);

}

