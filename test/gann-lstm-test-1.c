#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef WINDOWS

#else
#include <unistd.h>
#endif

#include "gann-lstm.h"

#define ITERATIONS  100000000
#define NO_EPOCHS   0

gnn_lstm_t**        model_layers;
gnn_lstm_params_t   params;
set_t set;

static int write_output_directly_bytes = 0;
static char *read_network = NULL;
static char *seed = NULL;
static int store_after_training = 0;
static char save_model_folder_raw[256];
static char save_model_folder_json[256];

void store_the_net_layers(int signo)
{
  if ( SIGINT == signo ) {
    if ( model_layers != NULL ) {
      lstm_store(params.store_network_name_raw, &set,
      model_layers, params.layers);
      lstm_store_net_layers_as_json(model_layers, params.store_network_name_json, JSON_KEY_NAME_SET, &set, params.layers);
      printf("\nStored the net as: '%s'\nYou can use that file in the .html interface.\n",
      params.store_network_name_json );
      printf("The net in its raw format is stored as: '%s'.\nYou can use that with the -r flag \
to continue refining the weights.\n", params.store_network_name_raw);
    } else {
      printf("\nFailed to store the net!\n");
      exit(-1);
    }
  }

  exit(0);
  return;
}

int main(int argc, char *argv[])
{
  int c;
  unsigned int p = 0;
  unsigned int file_size = 0, sz = 0;
  int *X_train, *Y_train;
  FILE * fp;

  memset(&params, 0, sizeof(params));

  params.iterations = ITERATIONS;
  params.epochs = NO_EPOCHS;
  params.loss_moving_avg = LOSS_MOVING_AVG;
  params.learning_rate = STD_LEARNING_RATE;
  params.momentum = STD_MOMENTUM;
  params.lambda = STD_LAMBDA;
  params.softmax_temp = SOFTMAX_TEMP;
  params.mini_batch_size = MINI_BATCH_SIZE;
  params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
  params.learning_rate_decrease = 0.03;
  params.stateful = 0;
  params.beta1 = 0.9;
  params.beta2 = 0.999;
  params.gradient_fit = GRADIENTS_FIT;
  params.gradient_clip = GRADIENTS_CLIP;
  params.decrease_lr = DECREASE_LR;
  params.model_regularize = MODEL_REGULARIZE;
  params.layers = 3;
  params.neurons = 68;
  params.optimizer = OPTIMIZE_ADAM;
  // Interaction configuration with the training of the network
  params.print_progress = PRINT_PROGRESS;
  params.print_progress_iterations = PRINT_EVERY_X_ITERATIONS;
  params.print_progress_sample_output = PRINT_SAMPLE_OUTPUT;
  params.print_progress_to_file = PRINT_SAMPLE_OUTPUT_TO_FILE;
  params.print_progress_number_of_chars = NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING;
  params.print_sample_output_to_file_arg = PRINT_SAMPLE_OUTPUT_TO_FILE_ARG;
  params.print_sample_output_to_file_name = PRINT_SAMPLE_OUTPUT_TO_FILE_NAME;
  params.store_progress_every_x_iterations = STORE_PROGRESS_EVERY_X_ITERATIONS;
  params.store_progress_file_name = PROGRESS_FILE_NAME;
  params.store_network_name_raw = STD_LOADABLE_NET_NAME;
  params.store_network_name_json = STD_JSON_NET_NAME;
  params.store_char_indx_map_name = JSON_KEY_NAME_SET;

  srand(time(NULL));

  parse_input_args(argc, argv);

  initialize_set(&set);

  fp = fopen(argv[1], "r");
  if ( fp == NULL )
  {
    printf("Could not open file: %s\n", argv[1]);
    return -1;
  }

  while ((c = fgetc(fp)) != EOF)
  {
    set_insert_symbol(&set, (char)c);
    ++file_size;
  }

  fclose(fp);

  X_train = calloc(file_size + 1, sizeof(int));
  if (X_train == NULL)
    return -1;

  X_train[file_size] = X_train[0];

  Y_train = &X_train[1];

  fp = fopen(argv[1], "r");
  while ((c = fgetc(fp)) != EOF)
    X_train[sz++] = set_char_to_indx(&set,c);
  fclose(fp);

  if ( read_network != NULL ) {
    int FRead;
    int FReadNewAfterDataFile;

    initialize_set(&set);

    lstm_load(read_network, &set, &params, &model_layers);

    if ( seed == NULL ) {

      FRead = set_get_features(&set);

      // Read from datafile, see if new features appear

      fp = fopen(argv[1], "r");
      if ( fp == NULL ) {
        printf("Could not open file: %s\n", argv[1]);
        return -1;
      }

      while ( ( c = fgetc(fp) ) != EOF ) {
        set_insert_symbol(&set, (char)c );
      }

      fclose(fp);

      FReadNewAfterDataFile = set_get_features(&set);

      if ( FReadNewAfterDataFile > FRead ) {
        // New features appeared. Must change
        // first and last layer.
        printf("New features detected in datafile.\nLoaded network worked with %d features\
, now there is %d features in total.\n\
Reallocating space in network input and output layer to accommodate this new feature set.\n",
          FRead, FReadNewAfterDataFile);

        lstm_reinit_model(
          model_layers,
          params.layers,
          FRead,
          FReadNewAfterDataFile
        );

      }

    }

    if ( seed == NULL )
      printf("Loaded the net: %s\n", read_network);
  } else {
    /* Allocating space for a new model */
    model_layers = calloc(params.layers, sizeof(gnn_lstm_t*));

    if ( model_layers == NULL ) {
      printf("Error in init!\n");
      exit(-1);
    }

    p = 0;
    while ( p < params.layers ) {
      // All layers have the same training parameters
      int X;
      int N = params.neurons;
      int Y;

      if ( params.layers == 1 ) {
        // 有多少个字符
        X = set_get_features(&set);
        Y = set_get_features(&set);
      } else {
        if ( p == 0 ) {
          // 第一层
          Y = set_get_features(&set);
          X = params.neurons;
        } else if ( p == params.layers - 1 ) {
          // 中间层
          Y = params.neurons;
          X = set_get_features(&set);
        } else {
          // 最后一层
          Y = params.neurons;
          X = params.neurons;
        }
      }

      model_layers[p] = gnn_lstm_new(X, N, Y, 0, &params);

      ++p;
    }
  }

  if (write_output_directly_bytes && read_network != NULL) {

    lstm_output_string_layers(model_layers, &set, set_indx_to_char(&set, 0), write_output_directly_bytes, params.layers);

    free(model_layers);
    free(X_train);
    return 0;
  } else if ( write_output_directly_bytes && read_network == NULL ) {
    usage(argv);
  }

  if ( seed != NULL ) {
    // output directly
    lstm_output_string_from_string(model_layers, &set, seed, params.layers, 256);

  } else {
    double loss;

    assert(params.layers > 0);

    printf("LSTM Neural net compiled: %s %s, %u Layers, ",
      __DATE__, __TIME__, params.layers);
    // Print neurons in each layer
    printf("Neurons: [");
    p = 0;
    while ( p < params.layers ) {
      printf("%s%d", (p>0?", ":""), model_layers[p]->N);
      ++p;
    }
    printf("], Features: %d.\n", model_layers[params.layers-1]->X);
    printf("Allocated bytes for the network: %s\n", prettyPrintBytes(e_alloc_total()));
    printf("Training parameters: Backprop Through Time: %d, LR: %lf, Mo: %lf, LA: %lf, LR-decrease: %lf.\n",
      MINI_BATCH_SIZE, params.learning_rate, params.momentum, params.lambda, params.learning_rate_decrease);

    signal(SIGINT, store_the_net_layers);

    gnn_lstm_train(model_layers,
                   &params,
                   &set,
                   file_size,
                   X_train,
                   Y_train,
                   params.layers,
                   &loss);

    if (store_after_training) {
      lstm_store(params.store_network_name_raw, &set,
      model_layers, params.layers);
      lstm_store_net_layers_as_json(model_layers, params.store_network_name_json,
        JSON_KEY_NAME_SET, &set, params.layers);
    }

    printf("Loss after training: %lf\n", loss);
  }

  free(model_layers);
  free(X_train);

  return 0;
}
