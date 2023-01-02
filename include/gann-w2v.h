/*!
**   .oooooo.          .o.       ooooo      ooo ooooo      ooo
**  d8P'  `Y8b        .888.      `888b.     `8' `888b.     `8'
** 888               .8"888.      8 `88b.    8   8 `88b.    8
** 888              .8' `888.     8   `88b.  8   8   `88b.  8
** 888     ooooo   .88ooo8888.    8     `88b.8   8     `88b.8
** `88.    .88'   .8'     `888.   8       `888   8       `888
**  `Y8bood8P'   o88o     o8888o o8o        `8  o8o        `8
*/
#ifndef __GANN_W2V_H__
#define __GANN_W2V_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <gfc.h>

#include "gann.h"

#define GANN_W2V_MAX_STRING                    100
#define GANN_W2V_EXP_TABLE_SIZE                1000
#define GANN_W2V_MAX_EXP                       6
#define GANN_W2V_MAX_SENTENCE_LENGTH           1000
#define GANN_W2V_MAX_CODE_LENGTH               4000
#define GANN_W2V_MIN_CHINESE                   0x4E00
#define GANN_W2V_MAX_CHINESE                   0x9FA5

/*!
** the word2vec neural network
*/
typedef struct gnn_w2v_s
{
  /*!
  ** the number of dimensions
  */
  uint        dim_num;

  /*!
  ** the size of vocabulary words
  */
  uint        vocab_size;

  /*!
  **
  */
  real*      hidden_weights;

  /*!
  **
  */
  real*      output_weights;

  /*!
  ** the hidden neurons, and the word2vec just has one hidden layer,
  ** and the size = vocab_size * dim_num
  */
  real*      hidden_neurons;

  /*!
  ** the softmax neurons, and the size = vocab_size * dim_num
  */
  real*      softmax_neurons;

  /*!
  ** the negative samplings, and the size = vocab_size * dim_num
  */
  real*      negative_samplings;

  /*!
  ** the size of distinct characters.
  */
  uint        char_size;

  /*!
  ** the character weights, the size = char_size * dim_num
  */
  float*      char_weights;

  uint*       embedded_count;

  uint*       last_embedded_count;
}
gnn_w2v_t;

typedef struct gnn_w2v_word_s
{
  ullong      count;

  int*        point;

  int*        character;
  int         character_size;
  int*        character_emb_select;

  char*       word;

  int         utf8len;

  int         len;

  char*       code;

  char        codelen;

  float*      weights;

  uint        index;

}
gnn_w2v_word_t;

typedef struct gnn_w2v_vocab_s
{
  /*!
  ** the distince words reading from file
  */
  gnn_w2v_word_t*       words;

  /*!
  ** the hash value for each word
  */
  uint*                 hashes;

  /*!
  ** the size of words
  */
  llong                 size;

  /*!
  ** the size of characters
  */
  llong                 char_size;

  int*                  unigram;
}
gnn_w2v_vocab_t;


/*!
**
*/
void
gnn_w2v_word_read(char* word, FILE* fin);

int
gnn_w2v_word_hash(const char* word);

int
gnn_w2v_word_index(gnn_w2v_vocab_t* vocab, const char* word);


/*!
** build
*/
void
gnn_w2v_unigram_build(gnn_w2v_vocab_t* vocab);

int
gnn_w2v_vocab_add(gnn_w2v_vocab_t* vocab, char *word, int is_non_comp);

void
gnn_w2v_vocab_sort(gnn_w2v_vocab_t* vocab);

/*!
**
*/
gnn_w2v_vocab_t*
gnn_w2v_read(const char* train_file_path);

/*!
**
*/
void
gnn_w2v_train(gnn_w2v_vocab_t*      vocab,
              uint                  dims,
              uint                  layer_size,
              const char*           model_file_path);

gnn_w2v_t*
gnn_w2v_build(gnn_w2v_vocab_t* vocab,
              uint dimensions);

void
gnn_w2v_skipgram(const char*            text_path,
                 gnn_w2v_vocab_t*       vocab,
                 uint                   sample,
                 uint                   window);

#ifdef __cplusplus
}
#endif

#endif // __GANN_W2V_H__
