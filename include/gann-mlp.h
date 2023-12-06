/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#ifndef __GNN_MLP_H__
#define __GNN_MLP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>

#ifndef GNN_MLP_RANDOM
/* We use the following for uniform random numbers between 0 and 1.
 * If you have a better function, redefine this macro. */
#define GNN_MLP_RANDOM() (((float)rand())/RAND_MAX)
#endif

typedef float (*gnn_mlp_activate)(float a);

struct gnn_mlp_s;

typedef struct gnn_mlp_s {

  /*!
  ** how many inputs, outputs, and hidden neurons.
  */
  int                   input_number, hidden_layer_number, hidden_neuron_number, output_number;

  /*!
  ** which activation function to use for hidden neurons. Default: gennann_act_sigmoid_cached
  */
  gnn_mlp_activate      activation_hidden;

  /*!
  ** which activation function to use for output. Default: gennann_act_sigmoid_cached
  */
  gnn_mlp_activate      activation_output;

  /*!
  ** total number of weights, and size of weights buffer.
  */
  int                   total_weights;

  /*!
  ** total number of neurons + inputs and size of output buffer.
  */
  int                   total_neurons;

  /*!
  ** all weights (total_weights).
  */
  float*                weights;

  /*!
  ** stores input array and output of each neuron (total_neurons).
  */
  float*                outputs;

  /*!
  ** stores delta of each hidden and output neuron (total_neurons - inputs).
  */
  float*                biases;

}
gnn_mlp_t;

static float MAGICAL_WEIGHT_NUMBER = 1.0f;
static float MAGICAL_LEARNING_NUMBER = 0.4f;


gnn_mlp_t *
gnn_mlp_new(int inputs, int hidden_layers, int hidden, int outputs);

void
gnn_mlp_free(gnn_mlp_t* mlp);

/*!
** normally runs mlp network to get output result, and it completes forward propagation too.
**
** @param mlp
**        the mlp network instance
**
** @param inputs
**        the input array
**
** @return the output result
*/
float const*
gnn_mlp_forward(gnn_mlp_t const* mlp,
                float const* inputs);

void
gnn_mlp_train(gnn_mlp_t   const*        mlp,
              float       const*        inputs,
              float       const*        desired_outputs,
              float                    learning_rate);

gnn_mlp_t*
gnn_mlp_read(FILE* in);

void
gnn_mlp_write(gnn_mlp_t const* mlp, FILE* out);


#ifdef __cplusplus
}
#endif

#endif
