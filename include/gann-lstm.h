/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#ifndef __GNN_LSTM_H__
#define __GNN_LSTM_H__

#ifdef __cplusplus
extern "C" {
#endif

#define OPTIMIZE_ADAM                       0
#define OPTIMIZE_GRADIENT_DESCENT           1

typedef struct gnn_lstm_params_s
{
  // For progress monitoring
  double loss_moving_avg;
  // For gradient descent
  double learning_rate;
  double momentum;
  double lambda;
  double softmax_temp;
  double beta1;
  double beta2;
  int gradient_clip;
  int gradient_fit;
  int optimizer;
  int model_regularize;
  int stateful;
  int decrease_lr;
  double learning_rate_decrease;

  /*!
  ** how many layers
  */
  uint  layers;

  /*!
  ** how many neurons this layer has
  */
  uint  neurons;

  // Output configuration for interactivity
  long  print_progress_iterations;
  int   print_progress_sample_output;
  int   print_progress;
  int   print_progress_to_file;
  int   print_progress_number_of_chars;
  char* print_sample_output_to_file_name;
  char* print_sample_output_to_file_arg;
  int   store_progress_every_x_iterations;
  char* store_progress_file_name;
  int   store_network_every;
  char* store_network_name_raw;
  char* store_network_name_json;
  char* store_char_indx_map_name;

  // General parameters
  unsigned int mini_batch_size;
  double gradient_clip_limit;
  unsigned long iterations;
  unsigned long epochs;
}
gnn_lstm_params_t;

typedef struct gnn_lstm_s
{
  /*!
  ** the number of input
  */
  unsigned int X;

  /*!
  ** the number of neurons
  */
  unsigned int N;

  /*!
  ** the number of output
  */
  unsigned int Y;

  /*!
  ** X + N
  */
  unsigned int S;

  // Parameters
  gnn_lstm_params_t* params;

  /*!
  ** the weights of forget gate
  */
  double* Wf;

  /*!
  ** the weights of input gate
  */
  double* Wi;

  /*!
  ** the weights of input node
  */
  double* Wc;

  /*!
  ** the weights of output gate
  */
  double* Wo;

  /*!
  ** the weights of output
  */
  double* Wy;

  /*!
  ** the bias of forget gate
  */
  double* bf;

  /*!
  ** the bias of input gate
  */
  double* bi;

  /*!
  ** the bias of input node
  */
  double* bc;

  /*!
  ** the bias of output gate
  */
  double* bo;

  /*!
  ** the bias of output
  */
  double* by;

  // descent layer hidden state
  double* dldh;
  double* dldho;
  double* dldhf;
  double* dldhi;
  double* dldhc;
  double* dldc;

  // descent layer input
  double* dldXi;
  double* dldXo;
  double* dldXf;
  double* dldXc;

  // gradient descent momentum
  double* Wfm;
  double* Wim;
  double* Wcm;
  double* Wom;
  double* Wym;
  double* bfm;
  double* bim;
  double* bcm;
  double* bom;
  double* bym;
}
gnn_lstm_t;



#ifdef __cplusplus
}
#endif

#endif // __GNN_LSTM_H__
