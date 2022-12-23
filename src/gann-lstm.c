/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#include <stdlib.h>
#include <stdio.h>

#include "gann.h"
#include "gann-lstm.h"

typedef struct gnn_lstm_values_cache_s
{
  double* probs;
  double* probs_before_sigma;
  double* c;
  double* h;
  double* c_old;
  double* h_old;
  double* X;

  /*!
  ** the weights of forgate gate in hidden state
  */
  double* hf;

  /*!
  ** the weights of input gate in hidden state
  */
  double* hi;

  /*!
  ** the weights of output gate in hidden state
  */
  double* ho;

  /*!
  ** the weights of input node in hidden state
  */
  double* hc;

  double* tanh_c_cache;
}
gnn_lstm_values_cache_t;

typedef struct gnn_lstm_values_state_s {
  double* c;
  double* h;
}
gnn_lstm_values_state_t;

typedef struct gnn_lstm_values_next_cache_s {
  double* dldh_next;
  double* dldc_next;
  double* dldY_pass;
}
gnn_lstm_values_next_cache_t;

void
gnn_lstm_forward_propagate(gnn_lstm_t*                model,
                           double*                    input,
                           gnn_lstm_values_cache_t*   cache_in,
                           gnn_lstm_values_cache_t*   cache_out,
                           int                        softmax)
{
  int N, Y, S, i = 0;
  double *h_old, *c_old, *X_one_hot;

  h_old = cache_in->h;
  c_old = cache_in->c;

  N = model->N;
  Y = model->Y;
  S = model->S;

#ifdef WINDOWS
  // MSVC is not a C99 compiler, and does not support variable length arrays
  // MSVC is documented as conforming to C90
  double *tmp;
  if ( init_zero_vector(&tmp, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n",
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }
#else
  double tmp[N]; // VLA must be supported.. May cause portability problems.. If so use init_zero_vector (will be slower).
#endif

  gnn_vec_copy(cache_out->h_old, h_old, N);
  gnn_vec_copy(cache_out->c_old, c_old, N);

  X_one_hot = cache_out->X;

  while ( i < S )
  {
    if ( i < N )
      X_one_hot[i] = h_old[i];
    else
      X_one_hot[i] = input[i - N];
    ++i;
  }

  /*!
  ** hf_t = wf_t * X_t + bf_t
  */
  gnn_lstm_full_forward(cache_out->hf, model->Wf, X_one_hot, model->bf, N, S);
  gnn_lstm_sigmoid_forward(cache_out->hf, cache_out->hf, N);

  gnn_lstm_full_forward(cache_out->hi, model->Wi, X_one_hot, model->bi, N, S);
  gnn_lstm_sigmoid_forward(cache_out->hi, cache_out->hi, N);

  gnn_lstm_full_forward(cache_out->ho, model->Wo, X_one_hot, model->bo, N, S);
  gnn_lstm_sigmoid_forward(cache_out->ho, cache_out->ho, N);

  gnn_lstm_full_forward(cache_out->hc, model->Wc, X_one_hot, model->bc, N, S);
  gnn_lstm_tanh_forward(cache_out->hc, cache_out->hc, N);

  /*!
  ** c = hf * c_old + hi * hc
  */
  gnn_vec_copy(cache_out->c, cache_out->hf, N);
  gnn_vec_multiply(cache_out->c, c_old, N);

  gnn_vec_copy(tmp, cache_out->hi, N);
  gnn_vec_multiply(tmp, cache_out->hc, N);

  gnn_vec_add(cache_out->c, tmp, N);

  /*!
  ** h = ho * tanh_c_cache
  */
  gnn_lstm_tanh_forward(cache_out->tanh_c_cache, cache_out->c, N);
  gnn_vec_copy(cache_out->h, cache_out->ho, N);
  gnn_vec_multiply(cache_out->h, cache_out->tanh_c_cache, N);

  /*!
  ** probs = softmax ( Wy*h + by )
  */
  gnn_lstm_full_forward(cache_out->probs, model->Wy, cache_out->h, model->by, Y, N);
  if (softmax > 0)
  {
    gnn_lstm_softmax_forward(cache_out->probs, cache_out->probs, Y, model->params->softmax_temp);
  }
#ifdef INTERLAYER_SIGMOID_ACTIVATION
  if (softmax <= 0)
  {
    gnn_lstm_sigmoid_forward(cache_out->probs, cache_out->probs, Y);
    gnn_vec_copy(cache_out->probs_before_sigma, cache_out->probs, Y);
  }
#endif

  gnn_vec_copy(cache_out->X, X_one_hot, S);

#ifdef WINDOWS
  free_vector(&tmp);
#endif

}

//              model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void
gnn_lstm_backward_propagate(gnn_lstm_t*                     model,
                            double*                         y_probabilities,
                            int                             y_correct,
                            gnn_lstm_values_next_cache_t*   d_next,
                            gnn_lstm_values_cache_t*        cache_in,
                            gnn_lstm_t*                     gradients,
                            gnn_lstm_values_next_cache_t*   cache_out)
{
  double *h,*dldh_next,*dldc_next, *dldy, *dldh, *dldho, *dldhf, *dldhi, *dldhc, *dldc;
  int N, Y, S;

  N = model->N;
  Y = model->Y;
  S = model->S;

  // model cache
  dldh = model->dldh;
  dldc = model->dldc;
  dldho = model->dldho;
  dldhi = model->dldhi;
  dldhf = model->dldhf;
  dldhc = model->dldhc;

  h = cache_in->h;

  dldh_next = d_next->dldh_next;
  dldc_next = d_next->dldc_next;

  dldy = y_probabilities;

  if ( y_correct >= 0 ) {
    dldy[y_correct] -= 1.0;
  }
#ifdef INTERLAYER_SIGMOID_ACTIVATION
  if ( y_correct < 0 ) {
    gnn_lstm_sigmoid_backward(dldy, cache_in->probs_before_sigma, dldy, Y);
  }
#endif

  gnn_lstm_full_backward(dldy, model->Wy, h, gradients->Wy, dldh, gradients->by, Y, N);
  gnn_vec_add(dldh, dldh_next, N);

  gnn_vec_copy(dldho, dldh, N);
  gnn_vec_multiply(dldho, cache_in->tanh_c_cache, N);
  gnn_lstm_sigmoid_backward(dldho, cache_in->ho, dldho, N);

  gnn_vec_copy(dldc, dldh, N);
  gnn_vec_multiply(dldc, cache_in->ho, N);
  gnn_lstm_tanh_backward(dldc, cache_in->tanh_c_cache, dldc, N);
  gnn_vec_add(dldc, dldc_next, N);

  gnn_vec_copy(dldhf, dldc, N);
  gnn_vec_multiply(dldhf, cache_in->c_old, N);
  gnn_lstm_sigmoid_backward(dldhf, cache_in->hf, dldhf, N);

  gnn_vec_multiply(dldhi, cache_in->hc, N);
  gnn_vec_multiply(dldhi, dldc, N);
  gnn_lstm_sigmoid_backward(dldhi, cache_in->hi, dldhi, N);

  gnn_vec_copy(dldhc, cache_in->hi, N);
  gnn_vec_multiply(dldhc, dldc, N);
  gnn_lstm_tanh_backward(dldhc, cache_in->hc, dldhc, N);

  gnn_lstm_full_backward(dldhi, model->Wi, cache_in->X, gradients->Wi, gradients->dldXi, gradients->bi, N, S);
  gnn_lstm_full_backward(dldhc, model->Wc, cache_in->X, gradients->Wc, gradients->dldXc, gradients->bc, N, S);
  gnn_lstm_full_backward(dldho, model->Wo, cache_in->X, gradients->Wo, gradients->dldXo, gradients->bo, N, S);
  gnn_lstm_full_backward(dldhf, model->Wf, cache_in->X, gradients->Wf, gradients->dldXf, gradients->bf, N, S);

  // dldXi will work as a temporary substitute for dldX (where we get extract dh_next from!)
  gnn_vec_add(gradients->dldXi, gradients->dldXc, S);
  gnn_vec_add(gradients->dldXi, gradients->dldXo, S);
  gnn_vec_add(gradients->dldXi, gradients->dldXf, S);

  gnn_vec_copy(cache_out->dldh_next, gradients->dldXi, N);
  gnn_vec_copy(cache_out->dldc_next, cache_in->hf, N);
  gnn_vec_multiply(cache_out->dldc_next, dldc, N);

  // To pass on to next layer
  gnn_vec_copy(cache_out->dldY_pass, &gradients->dldXi[N], model->X);
}

void lstm_zero_the_model(gnn_lstm_t * model)
{
  vector_set_to_zero(model->Wy, model->Y * model->N);
  vector_set_to_zero(model->Wi, model->N * model->S);
  vector_set_to_zero(model->Wc, model->N * model->S);
  vector_set_to_zero(model->Wo, model->N * model->S);
  vector_set_to_zero(model->Wf, model->N * model->S);

  vector_set_to_zero(model->by, model->Y);
  vector_set_to_zero(model->bi, model->N);
  vector_set_to_zero(model->bc, model->N);
  vector_set_to_zero(model->bf, model->N);
  vector_set_to_zero(model->bo, model->N);

  vector_set_to_zero(model->Wym, model->Y * model->N);
  vector_set_to_zero(model->Wim, model->N * model->S);
  vector_set_to_zero(model->Wcm, model->N * model->S);
  vector_set_to_zero(model->Wom, model->N * model->S);
  vector_set_to_zero(model->Wfm, model->N * model->S);

  vector_set_to_zero(model->bym, model->Y);
  vector_set_to_zero(model->bim, model->N);
  vector_set_to_zero(model->bcm, model->N);
  vector_set_to_zero(model->bfm, model->N);
  vector_set_to_zero(model->bom, model->N);

  vector_set_to_zero(model->dldhf, model->N);
  vector_set_to_zero(model->dldhi, model->N);
  vector_set_to_zero(model->dldhc, model->N);
  vector_set_to_zero(model->dldho, model->N);
  vector_set_to_zero(model->dldc, model->N);
  vector_set_to_zero(model->dldh, model->N);

  vector_set_to_zero(model->dldXc, model->S);
  vector_set_to_zero(model->dldXo, model->S);
  vector_set_to_zero(model->dldXi, model->S);
  vector_set_to_zero(model->dldXf, model->S);
}

/*!
** makes a new lstm network instance.
**
** @param X
**        the input number
**
** @param N
**        the hidden neuron number
**
** @param Y
**        the output number
*/
gnn_lstm_t*
gnn_lstm_new(int                  X,
             int                  N,
             int                  Y,
             int                  zeros,
             gnn_lstm_params_t*   params)
{
  int S = X + N;
  gnn_lstm_t* ret = e_calloc(1, sizeof(gnn_lstm_t));

  ret->X = X;
  ret->N = N;
  ret->S = S;
  ret->Y = Y;

  ret->params = params;

  if (zeros) {
    ret->Wf = gnn_vec_new(N * S, 0);
    ret->Wi = gnn_vec_new(N * S, 0);
    ret->Wc = gnn_vec_new(N * S, 0);
    ret->Wo = gnn_vec_new(N * S, 0);
    ret->Wy = gnn_vec_new(Y * N, 0);
  } else {
    ret->Wf = gnn_vec_new(N * S, (double)S);
    ret->Wi = gnn_vec_new(N * S, (double)S);
    ret->Wc = gnn_vec_new(N * S, (double)S);
    ret->Wo = gnn_vec_new(N * S, (double)S);
    ret->Wy = gnn_vec_new(Y * N, (double)N);
  }

  ret->bf = gnn_vec_new(N, 0);
  ret->bi = gnn_vec_new(N, 0);
  ret->bc = gnn_vec_new(N, 0);
  ret->bo = gnn_vec_new(N, 0);
  ret->by = gnn_vec_new(Y, 0);

  ret->dldhf = gnn_vec_new(N, 0);
  ret->dldhi = gnn_vec_new(N, 0);
  ret->dldhc = gnn_vec_new(N, 0);
  ret->dldho = gnn_vec_new(N, 0);
  ret->dldc  = gnn_vec_new(N, 0);
  ret->dldh  = gnn_vec_new(N, 0);

  ret->dldXc = gnn_vec_new(S, 0);
  ret->dldXo = gnn_vec_new(S, 0);
  ret->dldXi = gnn_vec_new(S, 0);
  ret->dldXf = gnn_vec_new(S, 0);

  // Gradient descent momentum caches
  ret->Wfm = gnn_vec_new(N * S, 0);
  ret->Wim = gnn_vec_new(N * S, 0);
  ret->Wcm = gnn_vec_new(N * S, 0);
  ret->Wom = gnn_vec_new(N * S, 0);
  ret->Wym = gnn_vec_new(Y * N, 0);

  ret->bfm = gnn_vec_new(N, 0);
  ret->bim = gnn_vec_new(N, 0);
  ret->bcm = gnn_vec_new(N, 0);
  ret->bom = gnn_vec_new(N, 0);
  ret->bym = gnn_vec_new(Y, 0);

  return ret;
}

/*!
**
** @param training_points
**        相当于训练数据中的字符个数，即训练数据的文件大小，标识了X_train的长度,
**
** @param X_train
**        训练文本集合
**
** @param Y_train
**        =X_train
*/
void
gnn_lstm_train(gnn_lstm_t**        model_layers,
               gnn_lstm_params_t*  params,
               set_t*              char_index_mapping,
               uint                training_points,
               int*                X_train,
               int*                Y_train,
               uint                layers,
               double*             loss_out)
{
  unsigned int p, i = 0, b = 0, q = 0, e1 = 0, e2 = 0,
    e3, record_iteration = 0, tmp_count, trailing;
  unsigned long n = 0, epoch = 0;
  double loss = -1, loss_tmp = 0.0, record_keeper = 0.0;
  double initial_learning_rate = params->learning_rate;
  time_t time_iter;
  char time_buffer[40];
  unsigned long iterations = params->iterations;
  unsigned long epochs = params->epochs;
  int stateful = params->stateful, decrease_lr = params->decrease_lr;
  // configuration for output printing during training
  int print_progress = params->print_progress;
  int print_progress_iterations = params->print_progress_iterations;
  int print_progress_sample_output = params->print_progress_sample_output;
  int print_progress_to_file = params->print_progress_to_file;
  int print_progress_number_of_chars = params->print_progress_number_of_chars;
  char *print_progress_to_file_name = params->print_sample_output_to_file_name;
  char *print_progress_to_file_arg = params->print_sample_output_to_file_arg;
  int store_progress_every_x_iterations = params->store_progress_every_x_iterations;
  char *store_progress_file_name = params->store_progress_file_name;
  int store_network_every = params->store_network_every;

  gnn_lstm_values_state_t**       stateful_d_next = NULL;
  gnn_lstm_values_cache_t***      cache_layers;
  gnn_lstm_values_next_cache_t**  d_next_layers;

  gnn_lstm_t**  gradient_layers = NULL;
  gnn_lstm_t**  gradient_layers_entry = NULL;
  gnn_lstm_t**  M_layers = NULL;
  gnn_lstm_t**  R_layers = NULL;

#ifdef WINDOWS
  double *first_layer_input = malloc(model_layers[0]->Y*sizeof(double));

  if ( first_layer_input == NULL ) {
    fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
      __FILE__, __func__, __LINE__, model_layers[0]->Y*sizeof(double));
    exit(1);
  }
#else
  double first_layer_input[model_layers[0]->Y];
#endif

  if ( stateful )
  {
    stateful_d_next = e_calloc(layers, sizeof(gnn_lstm_values_cache_t*));

    i = 0;
    while ( i < layers )
    {
      stateful_d_next[i] = e_calloc( training_points/params->mini_batch_size + 1, sizeof(gnn_lstm_values_state_t));

      lstm_values_state_init(&stateful_d_next[i], model_layers[i]->N);
      ++i;
    }
  }

  i = 0;
  cache_layers = e_calloc(layers, sizeof(gnn_lstm_values_cache_t**));

  while ( i < layers )
  {
    cache_layers[i] = e_calloc(params->mini_batch_size + 1, sizeof(gnn_lstm_values_cache_t*));

    p = 0;
    while (p < params->mini_batch_size + 1)
    {
      cache_layers[i][p] = lstm_cache_container_init(
        model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y);
      if ( cache_layers[i][p] == NULL )
        lstm_init_fail("Failed to allocate memory for the caches\n");
      ++p;
    }

    ++i;
  }

  gradient_layers = e_calloc(layers, sizeof(gnn_lstm_t*) );

  gradient_layers_entry = e_calloc(layers, sizeof(gnn_lstm_t*) );

  d_next_layers = e_calloc(layers, sizeof(gnn_lstm_values_next_cache_t *));

  if ( params->optimizer == OPTIMIZE_ADAM )
  {
    M_layers = e_calloc(layers, sizeof(gnn_lstm_t*) );
    R_layers = e_calloc(layers, sizeof(gnn_lstm_t*) );

  }

  i = 0;
  while (i < layers)
  {
    gradient_layers[i] = gnn_lstm_new(model_layers[i]->X,
                                      model_layers[i]->N,
                                      model_layers[i]->Y,
                                      1,
                                      params);
    gradient_layers_entry[i] = gnn_lstm_new(model_layers[i]->X,
                                            model_layers[i]->N,
                                            model_layers[i]->Y,
                                            1,
                                            params);

    lstm_values_next_cache_init(&d_next_layers[i],
      model_layers[i]->N, model_layers[i]->X);

    if ( params->optimizer == OPTIMIZE_ADAM )
    {
      M_layers[i] = gnn_lstm_new(model_layers[i]->X,
                                 model_layers[i]->N,
                                 model_layers[i]->Y,
                                 1,
                                 params);
      R_layers[i] = gnn_lstm_new(model_layers[i]->X,
                                 model_layers[i]->N,
                                 model_layers[i]->Y,
                                 1,
                                 params);
    }

    ++i;
  }

  /*!
  ** 开始训练
  */
  i = 0; b = 0;
  while ( n < iterations )
  {

    if ( epochs && epoch >= epochs ) break;

    b = i;

    loss_tmp = 0.0;

    q = 0;

    while ( q < layers )
    {
      if ( stateful ) {
        if ( q == 0 )
          lstm_cache_container_set_start(cache_layers[q][0],  model_layers[q]->N);
        else
          lstm_next_state_copy(stateful_d_next[q], cache_layers[q][0], model_layers[q]->N, 0);
      } else {
        lstm_cache_container_set_start(cache_layers[q][0], model_layers[q]->N);
      }
      ++q;
    }

    unsigned int check = i % training_points;

    trailing = params->mini_batch_size;

    if ( i + params->mini_batch_size >= training_points )
    {
      trailing = training_points - i;
    }

    q = 0;

    while ( q < trailing )
    {
      e1 = q;     // 当前这个
      e2 = q + 1; // 下一个

      e3 = i % training_points;

      tmp_count = 0;
      while ( tmp_count < model_layers[0]->Y )
      {
        first_layer_input[tmp_count] = 0.0;
        ++tmp_count;
      }

      first_layer_input[X_train[e3]] = 1.0;

      /* Layer numbering starts at the output point of the net */
      p = layers - 1;
      gnn_lstm_forward_propagate(model_layers[p],
                                 first_layer_input,
                                 cache_layers[p][e1],
                                 cache_layers[p][e2],
                                 p == 0);

      if ( p > 0 )
      {
        --p;
        while ( p <= layers - 1 )
        {
          gnn_lstm_forward_propagate(model_layers[p],
                                     cache_layers[p+1][e2]->probs,
                                     cache_layers[p][e1],
                                     cache_layers[p][e2],
                                     p == 0);
          --p;
        }
        p = 0;
      }

      loss_tmp += gnn_lstm_cross_entropy(cache_layers[p][e2]->probs, Y_train[e3]);
      ++i; ++q;
    }

    loss_tmp /= (q+1);

    if ( loss < 0 )
      loss = loss_tmp;

    loss = loss_tmp * params->loss_moving_avg + (1 - params->loss_moving_avg) * loss;

    if ( n == 0 )
      record_keeper = loss;

    if ( loss < record_keeper ) {
      record_keeper = loss;
      record_iteration = n;
    }

    if ( stateful ) {
      p = 0;
      while ( p < layers ) {
        lstm_next_state_copy(stateful_d_next[p], cache_layers[p][e2], model_layers[p]->N, 1);
        ++p;
      }
    }

    p = 0;
    while ( p < layers ) {
      lstm_zero_the_model(gradient_layers[p]);
      lstm_zero_d_next(d_next_layers[p], model_layers[p]->X, model_layers[p]->N);
      ++p;
    }

    while ( q > 0 ) {
      e1 = q;
      e2 = q - 1;

      e3 = ( training_points + i - 1 ) % training_points;

      p = 0;
      while ( p < layers ) {
        lstm_zero_the_model(gradient_layers_entry[p]);
        ++p;
      }

      p = 0;
      lstm_backward_propagate(model_layers[p],
        cache_layers[p][e1]->probs,
        Y_train[e3],
        d_next_layers[p],
        cache_layers[p][e1],
        gradient_layers_entry[0],
        d_next_layers[p]);

      if ( p < layers ) {
        ++p;
        while ( p < layers ) {
          lstm_backward_propagate(model_layers[p],
            d_next_layers[p-1]->dldY_pass,
            -1,
            d_next_layers[p],
            cache_layers[p][e1],
            gradient_layers_entry[p],
            d_next_layers[p]);
          ++p;
        }
      }

      p = 0;

      while ( p < layers ) {
        sum_gradients(gradient_layers[p], gradient_layers_entry[p]);
        ++p;
      }

      i--; q--;
    }

    assert(check == e3);

    p = 0;
    while ( p < layers ) {

      if ( params->gradient_clip )
        gradients_clip(gradient_layers[p], params->gradient_clip_limit);

      if ( params->gradient_fit )
        gradients_fit(gradient_layers[p], params->gradient_clip_limit);

      ++p;
    }

    p = 0;

    /*!
    ** 梯度下降优化
    */
    switch ( params->optimizer ) {
    case OPTIMIZE_ADAM:
      while ( p < layers ) {
        gradients_adam_optimizer(
          model_layers[p],
          gradient_layers[p],
          M_layers[p],
          R_layers[p],
          n);
        ++p;
      }
      break;
    case OPTIMIZE_GRADIENT_DESCENT:
      while ( p < layers ) {
        gradients_decend(model_layers[p], gradient_layers[p]);
        ++p;
      }
      break;
    default:
      fprintf( stderr,
        "Failed to update gradients, no acceptible optimization algorithm provided.\n\
        lstm_model_parameters_t has a field called 'optimizer'. Set this value to:\n\
        %d: Adam gradients optimizer algorithm\n\
        %d: Gradients descent algorithm.\n",
        OPTIMIZE_ADAM,
        OPTIMIZE_GRADIENT_DESCENT
      );
      exit(1);
      break;
    }

    if ( print_progress && !( n % print_progress_iterations ) ) {

      memset(time_buffer, '\0', sizeof time_buffer);
      time(&time_iter);
      strftime(time_buffer, sizeof time_buffer, "%X", localtime(&time_iter));

      printf("%s Iteration: %lu (epoch: %lu), Loss: %lf, record: %lf (iteration: %d), LR: %lf\n",
        time_buffer, n, epoch, loss, record_keeper, record_iteration, params->learning_rate);

      if ( print_progress_sample_output ) {
        printf("=====================================================\n");
        lstm_output_string_layers(model_layers, char_index_mapping, X_train[b],
          print_progress_number_of_chars, layers);
        printf("\n=====================================================\n");
      }

      if ( print_progress_to_file ) {
        FILE * fp_progress_output = fopen(print_progress_to_file_name,
          print_progress_to_file_arg);
        if ( fp_progress_output != NULL ) {
          fprintf(fp_progress_output, "%s====== Iteration: %lu, loss: %.5lf ======\n", n==0 ? "" : "\n", n, loss);
          lstm_output_string_layers_to_file(fp_progress_output, model_layers, char_index_mapping, X_train[b], print_progress_number_of_chars, layers);
          fclose(fp_progress_output);
        }
      }

      // Flushing stdout
      fflush(stdout);
    }

    if ( store_progress_every_x_iterations && !(n % store_progress_every_x_iterations ))
      lstm_store_progress(store_progress_file_name, n, loss);

    if ( store_network_every && !(n % store_network_every) ) {
      lstm_store(
        params->store_network_name_raw,
        char_index_mapping,
        model_layers,
        layers);
      lstm_store_net_layers_as_json(model_layers, params->store_network_name_json,
        params->store_char_indx_map_name, char_index_mapping, layers);
    }

    if ( b + params->mini_batch_size >= training_points )
      epoch++;

    i = (b + params->mini_batch_size) % training_points;

    if ( i < params->mini_batch_size ) {
      i = 0;
    }

    if ( decrease_lr ) {
      params->learning_rate = initial_learning_rate / ( 1.0 + n / params->learning_rate_decrease );
      //printf("learning rate: %lf\n", model->params->learning_rate);
    }

    ++n;
  }

  // Reporting the loss value
  *loss_out = loss;

  /*!
  ** 释放资源
  */
  p = 0;
  while ( p < layers ) {
    lstm_values_next_cache_free(d_next_layers[p]);

    i = 0;
    while ( i < params->mini_batch_size ) {
      lstm_cache_container_free(cache_layers[p][i]);
      lstm_cache_container_free(cache_layers[p][i]);
      ++i;
    }

    if ( params->optimizer == OPTIMIZE_ADAM ) {
      lstm_free_model(M_layers[p]);
      lstm_free_model(R_layers[p]);
    }

    lstm_free_model(gradient_layers_entry[p]);
    lstm_free_model(gradient_layers[p]);

    ++p;
  }

  if ( stateful && stateful_d_next != NULL ) {
    i = 0;
    while ( i < layers ) {
      free(stateful_d_next[i]);
      ++i;
    }
    free(stateful_d_next);
  }


  free(cache_layers);
  free(gradient_layers);
  if ( M_layers != NULL )
    free(M_layers);
  if ( R_layers != NULL )
    free(R_layers);
#ifdef WINDOWS
  free(first_layer_input);
#endif
}

/*!
** Y = AX + b
*/
void
gnn_lstm_full_forward(double*    Y,
                      double*    A,
                      double*    X,
                      double*    b,
                      int        R,
                      int        C)
{
  int i = 0, n = 0;
  while (i < R)
  {
    Y[i] = b[i];
    n = 0;
    while (n < C)
    {
      Y[i] += A[i * C + n] * X[n];
      ++n;
    }
    ++i;
  }
}

/*!
** Y = AX + b
*/
void
gnn_lstm_full_backward(double*   dldY,
                       double*   A,
                       double*   X,
                       double*   dldA,
                       double*   dldX,
                       double*   dldb,
                       int       R,
                       int       C)
{
  int i = 0, n = 0;

  // computing dldA
  while ( i < R ) {
    n = 0;
    while ( n < C ) {
      dldA[i * C + n] = dldY[i] * X[n];
      ++n;
    }
    ++i;
  }

  // computing dldb (easy peasy)
  i = 0;
  while ( i < R )
  {
    dldb[i] = dldY[i];
    ++i;
  }

  // computing dldX
  i = 0, n = 0;
  while ( i < C )
  {
    n = 0;

    dldX[i] = 0.0;
    while ( n < R ) {
      dldX[i] += A[n * C + i] * dldY[n];
      ++n;
    }

    ++i;
  }
}

double
gnn_lstm_cross_entropy(double* probabilities, int correct)
{
  return -log(probabilities[correct]);
}

// Dealing with softmax layer, forward and backward
//                &P,   Y,    features
void
gnn_lstm_softmax_forward(double* P,
                         double* Y,
                         int F,
                         double temperature)
{
  int f = 0;
  double sum = 0;
#ifdef WINDOWS
  // MSVC is not a C99 compiler, and does not support variable length arrays
  // MSVC is documented as conforming to C90
  double *cache = malloc(sizeof(double)*F);

  if ( cache == NULL ) {
    fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n",
      __FILE__, __func__, __LINE__, sizeof(double)*F);
    exit(1);
  }
#else
  double cache[F];
#endif

  while ( f < F ) {
    cache[f] = exp(Y[f] / temperature);
    sum += cache[f];
    ++f;
  }

  f = 0;
  while ( f < F ) {
    P[f] = cache[f] / sum;
    ++f;
  }

#ifdef WINDOWS
  free(cache);
#endif
}
//                    P,    c,  &dldh, rows
void
gnn_lstm_softmax_backward(double* P,
                          int c,
                          double* dldh,
                          int R)
{
  int r = 0;

  while ( r < R ) {
    dldh[r] = P[r];
    ++r;
  }

  dldh[c] -= 1.0;
}

/*!
** Y = sigmoid(X)
*/
void  gnn_lstm_sigmoid_forward(double* Y, double* X, int L)
{
  int l = 0;

  while ( l < L )
  {
    Y[l] = 1.0 / ( 1.0 + exp(-X[l]));
    ++l;
  }

}

/*!
** Y = sigmoid(X)
*/
void
gnn_lstm_sigmoid_backward(double* dldY, double* Y, double* dldX, int L)
{
  int l = 0;

  while ( l < L )
  {
    dldX[l] = ( 1.0 - Y[l] ) * Y[l] * dldY[l];
    ++l;
  }
}

/*!
** Y = tanh(X)
*/
void gnn_lstm_tanh_forward(double* Y, double* X, int L)
{
  int l = 0;
  while ( l < L ) {
    Y[l] = tanh(X[l]);
    ++l;
  }
}

/*!
** Y = tanh(X)
*/
void  gnn_lstm_tanh_backward(double* dldY, double* Y, double* dldX, int L)
{
  int l = 0;
  while ( l < L )
  {
    dldX[l] = ( 1.0 - Y[l] * Y[l] ) * dldY[l];
    ++l;
  }
}

