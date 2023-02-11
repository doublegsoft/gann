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
#include <stdlib.h>
#include <string.h>

#include <gfc.h>
#include <gnum.h>

#include "gann-w2v.h"

#define MAX_EXP             6
#define EXP_TABLE_SIZE      1000

typedef struct gnn_w2v_train_params_s
{
  float                 alpha;

  uint                  dimensions;

  ullong                sample;

  llong                 id;

  ullong                file_size;

  char                  file_path[4096];

  gnn_w2v_vocab_t*      vocab;
}
gnn_w2v_train_params_t;

// Maximum 30 * 0.7 = 21M words in the vocabulary
const int vocab_hash_size = 30000000;

long long vocab_max_size = 1000;

// unigram table size
const static int table_size = 1e8;

int cbow = 1, debug_mode = 2, window = 5, min_count = 0, num_threads = 12, min_reduce = 1;
int cwe_type = 2, multi_emb = 3, *embed_count, cwin = 5;

static int
gnn_w2v_word_compare(const void* a, const void* b)
{
  return ((gnn_w2v_word_t*)b)->count - ((gnn_w2v_word_t*)a)->count;
}

static void
gnn_w2v_vocab_huffman(gnn_w2v_vocab_t* vocab)
{
#ifdef DEBUG
  FILE* jsout = fopen("../../debug.json", "w");
#endif
  long long a, b, i, min1i, min2i, pos1, pos2, point[GANN_W2V_MAX_CODE_LENGTH];
  char code[GANN_W2V_MAX_CODE_LENGTH];
  long long* count = (long long *)calloc(vocab->size * 2 + 1, sizeof(long long));
  long long* binary = (long long *)calloc(vocab->size * 2 + 1, sizeof(long long));
  long long* parent_node = (long long *)calloc(vocab->size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab->size; a++)
  {
    count[a] = vocab->words[a].count;
#ifdef DEBUG
    fprintf(jsout, "SetCount(%d, %d, 0);\n", a, count[a]);
#endif
  }
  for (a = vocab->size; a < vocab->size * 2; a++)
  {
    count[a] = 1e15;
#ifdef DEBUG
    fprintf(jsout, "SetCount(%d, '-', 1);\n",a);
#endif
  }
  pos1 = vocab->size - 1;
  pos2 = vocab->size;
#ifdef DEBUG
  fprintf(jsout, "SetPos1(%d)\n",pos1);
  fprintf(jsout, "SetPos2(%d)\n",pos2);
#endif
  // following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab->size - 1; a++) {
    // first, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2])
      {
        min1i = pos1;
        pos1--;
#ifdef DEBUG
        fprintf(jsout, "SetMin1i(%d)\n",min1i);
        fprintf(jsout, "SetPos1(%d)\n",pos1);
#endif
      }
      else
      {
        min1i = pos2;
        pos2++;
#ifdef DEBUG
        fprintf(jsout, "SetMin1i(%d)\n",min1i);
        fprintf(jsout, "SetPos2(%d)\n",pos2);
#endif
      }
    } else {
      min1i = pos2;
      pos2++;
#ifdef DEBUG
      fprintf(jsout, "SetMin1i(%d)\n",min1i);
      fprintf(jsout, "SetPos2(%d)\n",pos2);
#endif
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
#ifdef DEBUG
        fprintf(jsout, "SetMin1i(%d)\n",min1i);
        fprintf(jsout, "SetPos2(%d)\n",pos2);
#endif
      }
      else
      {
        min2i = pos2;
        pos2++;
#ifdef DEBUG
        fprintf(jsout, "SetMin2i(%d)\n",min2i);
        fprintf(jsout, "SetPos2(%d)\n",pos2);
#endif
      }
    } else {
      min2i = pos2;
      pos2++;
#ifdef DEBUG
      fprintf(jsout, "SetMin2i(%d)\n",min2i);
      fprintf(jsout, "SetPos2(%d)\n",pos2);
#endif
    }
    count[vocab->size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab->size + a;
    parent_node[min2i] = vocab->size + a;
    binary[min2i] = 1;
#ifdef DEBUG
    fprintf(jsout, "SetCount(%d, %d, 2)\n", vocab->size + a, count[vocab->size + a]);
    fprintf(jsout, "SetParent(%d, %d)\n", min1i, parent_node[min1i]);
    fprintf(jsout, "SetParent(%d, %d)\n",min2i, parent_node[min2i]);
    fprintf(jsout, "SetBinary(%d, %d);\n\n", min2i, binary[min2i]);
#endif
  }

  for (a = 0; a < vocab->size; a++)
  {
    b = a;
    i = 0;
    while (1)
    {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab->size * 2 - 2) break;
    }
    vocab->words[a].codelen = i;
    vocab->words[a].point[0] = vocab->size - 2;
    for (b = 0; b < i; b++)
    {
      vocab->words[a].code[i - b - 1] = code[b];
      vocab->words[a].point[i - b] = point[b] - vocab->size;
    }
  }

  FILE* pout = fopen("../../parent.txt", "w");
  for (a = 0; a < vocab->size * 2 - 2; a++)
  {
    fprintf(pout, "%d = %lld = %lld\n", a, parent_node[a], count[a]);
  }
  fclose(pout);

  free(count);
  free(binary);
  free(parent_node);
#ifdef DEBUG
  fclose(jsout);
#endif
}

//static void
//gnn_w2v_cbow_train(gnn_w2v_vocab_t* vocab, float* neurons, int neuron_size)
//{
//
//  cw = 0;
//  char_list_cnt = 0;
//  for (a = b; a < sliding_window * 2 + 1 - b; a++) if (a != sliding_window) {
//    index = c = sentence_position - window + a;
//    if (c < 0) continue;
//    if (c >= sentence_length) continue;
//    last_word = sen[c];
//    if (last_word == -1) continue;
//    for (c = 0; c < dim; c++) neu1char[c] = 0;
//    for (c = 0; c < dim; c++) neu1char[c] = syn0[c + last_word * dim];
//    if (cwe_type && vocab[last_word].character_size) {
//      if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
//        get_base(base, sen, sentence_length, index);
//      }
//      for (c = 0; c < vocab[last_word].character_size; c++) {
//        charv_id = vocab[last_word].character[c];
//        if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
//          charv_id = get_emb(base, charv_id, last_word, c);
//        }
//        for (d = 0; d < dim; d++)
//          neu1char[d] += charv[d + charv_id * dim] / vocab[last_word].character_size;
//        charv_id_list[char_list_cnt] = charv_id;
//        char_list_cnt++;
//      }
//      for (d = 0; d < dim; d++) neu1char[d] /= 2;
//    }
//    for (c = 0; c < dim; c++) neu1[c] += neu1char[c];
//    cw++;
//  }
//  if (cw) {
//    for (c = 0; c < dim; c++) neu1[c] /= cw;
//    if (hs) for (d = 0; d < vocab[word].codelen; d++) {
//      f = 0;
//      l2 = vocab[word].point[d] * dim;
//      // Propagate hidden -> output
//      for (c = 0; c < dim; c++) f += neu1[c] * syn1[c + l2];
//      if (f <= -MAX_EXP) continue;
//      else if (f >= MAX_EXP) continue;
//      else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
//      // 'g' is the gradient multiplied by the learning rate
//      g = (1 - vocab[word].code[d] - f) * alpha;
//      // Propagate errors output -> hidden
//      for (c = 0; c < dim; c++) neu1e[c] += g * syn1[c + l2];
//      // Learn weights hidden -> output
//      for (c = 0; c < dim; c++) syn1[c + l2] += g * neu1[c];
//    }
//    // NEGATIVE SAMPLING
//    if (negative > 0) for (d = 0; d < negative + 1; d++) {
//      if (d == 0) {
//        target = word;
//        label = 1;
//      } else {
//        next_random = next_random * (unsigned long long)25214903917 + 11;
//        target = table[(next_random >> 16) % table_size];
//        if (target == 0) target = next_random % (vocab_size - 1) + 1;
//        if (target == word) continue;
//        label = 0;
//      }
//      l2 = target * dim;
//      f = 0;
//      for (c = 0; c < dim; c++) f += neu1[c] * syn1neg[c + l2];
//      if (f > MAX_EXP) g = (label - 1) * alpha;
//      else if (f < -MAX_EXP) g = (label - 0) * alpha;
//      else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
//      for (c = 0; c < dim; c++) neu1e[c] += g * syn1neg[c + l2];
//      for (c = 0; c < dim; c++) syn1neg[c + l2] += g * neu1[c];
//    }
//    // hidden -> in
//    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
//      c = sentence_position - window + a;
//      if (c < 0) continue;
//      if (c >= sentence_length) continue;
//      last_word = sen[c];
//      if (last_word == -1) continue;
//      for (c = 0; c < dim; c++) syn0[c + last_word * dim] += neu1e[c];
//    }
//    for (a = 0; a < char_list_cnt; a++) {
//      charv_id = charv_id_list[a];
//      for (c = 0; c < dim; c++) charv[c + charv_id * dim] += neu1e[c] * char_rate;
//    }
//  }
//}
//
//static void*
//gnn_w2v_vocab_train(void* data)
//{
//  gnn_w2v_train_params_t* params = (gnn_w2v_train_params_t*) data;
//  llong a, b, d, cw, t1, t2, word_index, last_word, sentence_length = 0, sentence_position = 0, charv_id;
//  llong word_count = 0, last_word_count = 0, sen[GANN_W2V_MAX_SENTENCE_LENGTH + 1];
//  llong l1, l2, c, target, label, local_iter = iter, index;
//  llong *charv_id_list = calloc(GANN_W2V_MAX_SENTENCE_LENGTH, sizeof(long long));
//  int char_list_cnt;
//  ullong next_random = params->id;
//  real f, g;
//  clock_t now;
//  real *neu1 = (real *)calloc(params->dimensions, sizeof(real));
//  real *neu1char = calloc(params->dimensions, sizeof(real));
//  real *neu1e = (real *)calloc(params->dimensions, sizeof(real));
//  real *base = (real *)calloc(params->dimensions, sizeof(real));
//  FILE *fi = fopen(params->file_path, "rb");
//  fseek(fi, params->file_size / (long long)num_threads * (long long)params->id, SEEK_SET);
//  while (1) {
//    if (word_count - last_word_count > 10000) {
//      word_count_actual += word_count - last_word_count;
//      last_word_count = word_count;
//      if ((debug_mode > 1)) {
//        now=clock();
//        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
//         word_count_actual / (real)(iter * train_words + 1) * 100,
//         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
//        fflush(stdout);
//      }
//      params->alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
//      if (params->alpha < starting_alpha * 0.0001)
//        params->alpha = starting_alpha * 0.0001;
//    }
//    if (sentence_length == 0) {
//      while (1)
//      {
//        word_index = gnn_w2v_word_index(fi);
//        if (feof(fi)) break;
//        if (word_index == -1) continue;
//        word_count++;
//        if (word_index == 0) break;
//        // The subsampling randomly discards frequent words while keeping the ranking same
//        if (params->sample > 0) {
//          real ran = (sqrt(params->vocab->words[word_index].count / (params->sample * train_words)) + 1)
//              * (params->sample * train_words) / params->vocab->words[word_index].count;
//          next_random = next_random * (unsigned long long)25214903917 + 11;
//          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
//        }
//        sen[sentence_length] = word_index;
//        sentence_length++;
//        if (sentence_length >= GANN_W2V_MAX_SENTENCE_LENGTH) break;
//      }
//      sentence_position = 0;
//    }
//    if (feof(fi) || (word_count > train_words / num_threads)) {
//      word_count_actual += word_count - last_word_count;
//      local_iter--;
//      if (local_iter == 0) break;
//      word_count = 0;
//      last_word_count = 0;
//      sentence_length = 0;
//      fseek(fi, params->file_size / (long long)num_threads * (long long) params->id, SEEK_SET);
//      continue;
//    }
//    word_index = sen[sentence_position];
//    if (word_index == -1) continue;
//    for (c = 0; c < params->dimensions; c++) neu1[c] = 0;
//    for (c = 0; c < params->dimensions; c++) neu1e[c] = 0;
//    for (c = 0; c < params->dimensions; c++) base[c] = 0;
//    next_random = next_random * (unsigned long long)25214903917 + 11;
//    b = next_random % window;
//
//
//    if (cbow)
//    {
//      /*!
//      ** train the cbow architecture
//      */
//
//    }
//    else
//    {
//      /*!
//      ** train the skip-gram architecture
//      */
//      for (a = b; a < window * 2 + 1 - b; a++)
//        if (a != window)
//        {
//        index = c = sentence_position - window + a;
//        if (c < 0) continue;
//        if (c >= sentence_length) continue;
//        last_word = sen[c];
//        if (last_word == -1) continue;
//        l1 = last_word * dim;
//        for (c = 0; c < dim; c++) neu1[c] = 0;
//        for (c = 0; c < dim; c++) neu1[c] = syn0[c + l1];
//        char_list_cnt = 0;
//        if (cwe_type && vocab[last_word].character_size) {
//          if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
//            get_base(base, sen, sentence_length, index);
//          }
//          for (c = 0; c < vocab[last_word].character_size; c++) {
//            charv_id = vocab[last_word].character[c];
//            if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
//              charv_id = get_emb(base, charv_id, last_word, c);
//            }
//            for (d = 0; d < dim; d++)
//              neu1[d] += charv[d + charv_id * dim] / vocab[last_word].character_size;
//            charv_id_list[char_list_cnt] = charv_id;
//            char_list_cnt++;
//          }
//          for (d = 0; d < dim; d++) neu1[d] /= 2;
//        }
//
//        for (c = 0; c < dim; c++) neu1e[c] = 0;
//        // HIERARCHICAL SOFTMAX
//        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
//          f = 0;
//          l2 = vocab[word].point[d] * dim;
//          // Propagate hidden -> output
//          for (c = 0; c < dim; c++) f += neu1[c] * syn1[c + l2];
//          if (f <= -MAX_EXP) continue;
//          else if (f >= MAX_EXP) continue;
//          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
//          // 'g' is the gradient multiplied by the learning rate
//          g = (1 - vocab[word].code[d] - f) * alpha;
//          // Propagate errors output -> hidden
//          for (c = 0; c < dim; c++) neu1e[c] += g * syn1[c + l2];
//          // Learn weights hidden -> output
//          for (c = 0; c < dim; c++) syn1[c + l2] += g * neu1[c];
//        }
//        // NEGATIVE SAMPLING
//        if (negative > 0) for (d = 0; d < negative + 1; d++) {
//          if (d == 0) {
//            target = word;
//            label = 1;
//          } else {
//            next_random = next_random * (unsigned long long)25214903917 + 11;
//            target = table[(next_random >> 16) % table_size];
//            if (target == 0) target = next_random % (vocab_size - 1) + 1;
//            if (target == word) continue;
//            label = 0;
//          }
//          l2 = target * dim;
//          f = 0;
//          for (c = 0; c < dim; c++) f += neu1[c] * syn1neg[c + l2];
//          if (f > MAX_EXP) g = (label - 1) * alpha;
//          else if (f < -MAX_EXP) g = (label - 0) * alpha;
//          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
//          for (c = 0; c < dim; c++) neu1e[c] += g * syn1neg[c + l2];
//          for (c = 0; c < dim; c++) syn1neg[c + l2] += g * neu1[c];
//        }
//        // Learn weights input -> hidden
//        for (c = 0; c < dim; c++) syn0[c + l1] += neu1e[c];
//        for (c = 0; c < char_list_cnt; c++) {
//          charv_id = charv_id_list[c];
//          for (d = 0; d < dim; d++) charv[d + charv_id * dim] += neu1e[d] * char_rate;
//        }
//      }
//    }
//    sentence_position++;
//    if (sentence_position >= sentence_length) {
//      sentence_length = 0;
//      continue;
//    }
//  }
//  fclose(fi);
//  free(neu1);
//  free(neu1e);
//  pthread_exit(NULL);
//}

/*!
**
*/
static int ignore = 0;

void
gnn_w2v_word_read(char* word, FILE* fin)
{
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == ':' && a == 0)
    {
      while (ch != '\n' && ch != '\r')
      {
        ch = fgetc(fin);
      }
    }
    // if (ch == '\r' || ch == '\n') continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n') || ch == '\r') {
      if (a > 0)
      {
        if (ch == '\n' || ch == '\r')
        {
          ignore = 0;
          ungetc(ch, fin);
        }
        break;
      }
      if (ch == '\n' || ch == '\r')
      {
        ignore = 0;
        return;
      }
      else continue;
    }
    word[a] = ch;
    a++;
    if (a >= GANN_W2V_MAX_STRING - 1) a--;   // Truncate too long words
  }
  if (ignore)
    word[0] = '\0';
  else
    word[a] = 0;
}

int
gnn_w2v_word_hash(const char* word)
{
  unsigned int len = strlen(word);
  unsigned long long a, hash = 0;
  for (a = 0; a < len; a++)
    hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

int
gnn_w2v_word_index(gnn_w2v_vocab_t* vocab, const char* word)
{
  unsigned int hash = gnn_w2v_word_hash(word);
  while (1)
  {
    if (vocab->hashes[hash] == -1)
      return -1;
    if (!strcmp(word, vocab->words[vocab->hashes[hash]].word))
    {
      return vocab->hashes[hash];
    }
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}


/*!
** build
*/
void
gnn_w2v_vocab_unigram(gnn_w2v_vocab_t* vocab)
{
  int a, i;
  long long train_words_pow = 0;
  float d1, power = 0.75;
  vocab->unigram = (int*) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab->size; a++)
    train_words_pow += pow(vocab->words[a].count, power);
  i = 0;
  d1 = pow(vocab->words[i].count, power) / (double)train_words_pow;
  for (a = 0; a < table_size; a++)
  {
    vocab->unigram[a] = i;
    if ((a / (float)table_size) > d1)
    {
      i++;
      d1 += pow(vocab->words[i].count, power) / (double)train_words_pow;
    }
    if (i >= vocab->size)
      i = vocab->size - 1;
  }
}

int
gnn_w2v_vocab_add(gnn_w2v_vocab_t* vocab, char* word, int is_non_comp) {
  unsigned int hash, length = strlen(word) + 1, len, i, pos;

  vocab->words[vocab->size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab->words[vocab->size].word, word);
  vocab->words[vocab->size].utf8len = gfc_utf8_length(word);
  vocab->words[vocab->size].len = strlen(word);
  vocab->words[vocab->size].count = 1;
  vocab->size++;

  // reallocate memory if needed
  if (vocab->size + 2 >= vocab_max_size)
  {
    vocab_max_size += 1000;
    vocab->words = (gnn_w2v_word_t *)realloc(vocab->words, vocab_max_size * sizeof(gnn_w2v_word_t));
  }
  hash = gnn_w2v_word_hash(word);
  while (vocab->hashes[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab->hashes[hash] = vocab->size - 1;
  return vocab->size - 1;
}

// Sorts the vocabulary by frequency using word counts
void
gnn_w2v_vocab_sort(gnn_w2v_vocab_t* vocab) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab->words[1], vocab->size - 1, sizeof(gnn_w2v_word_t), gnn_w2v_word_compare);
  for (a = 0; a < vocab_hash_size; a++) vocab->hashes[a] = -1;
  size = vocab->size;
  for (a = 0; a < size; a++)
  {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab->words[a].count < min_count) && (a != 0)) {
//      vocab->size--;
//      free(vocab->words[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = gnn_w2v_word_hash(vocab->words[a].word);
      while (vocab->hashes[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab->hashes[hash] = a;
    }
  }
  vocab->words = (gnn_w2v_word_t *) realloc(vocab->words, (vocab->size + 1) * sizeof(gnn_w2v_word_t));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab->size; a++)
  {
    vocab->words[a].code = (char *)calloc(GANN_W2V_MAX_CODE_LENGTH, sizeof(char));
    vocab->words[a].point = (int *)calloc(GANN_W2V_MAX_CODE_LENGTH, sizeof(int));
  }
}

gnn_w2v_vocab_t*
gnn_w2v_read(const char* train_file_path)
{
  char            word[GANN_W2V_MAX_STRING];
  FILE*           fin;
  llong           a, i;
  llong           train_words = 0;

  /*!
  ** 初始化词汇表, UTF-8编码
  */
  gnn_w2v_vocab_t* vocab = (gnn_w2v_vocab_t*) calloc(1, sizeof(gnn_w2v_vocab_t));
  vocab->words = (gnn_w2v_word_t *)calloc(vocab_max_size, sizeof(gnn_w2v_word_t));
  vocab->hashes = calloc(vocab_hash_size, sizeof(uint));
  vocab->size = 0;
  for (a = 0; a < vocab_hash_size; a++)
    vocab->hashes[a] = -1;

  fin = fopen(train_file_path, "rb");
  if (fin == NULL)
  {
    fprintf(stderr, "ERROR: training data file '%s' not found!\n", train_file_path);
    exit(1);
  }

  /*!
  ** 首个词汇永远是</s>
  */
  gnn_w2v_vocab_add(vocab, (char *)"</s>", 0);
//  if (strlen(non_comp)) LearnNonCompWord();

  while (1)
  {
    gnn_w2v_word_read(word, fin);
    // if encounter linefeed, the word is empty
    if (strlen(word) == 0 || word[0] == 1) continue;
    if (feof(fin)) break;

    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0))
    {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = gnn_w2v_word_index(vocab, (const char*) word);
    if (i == -1)
    {
      a = gnn_w2v_vocab_add(vocab, word, 0);
      vocab->words[a].count = 1;
    }
    else
      vocab->words[i].count++;
    word[0] = '\0';
//    if (vocab->size > vocab_hash_size * 0.7) ReduceVocab();
  }


#ifdef DEBUG
  fprintf(stdout, "Vocab size: %lld\n", vocab->size);
  fprintf(stdout, "Words in train file: %lld\n", train_words);
#endif
  fclose(fin);

  gnn_w2v_vocab_unigram(vocab);
  gnn_w2v_vocab_sort(vocab);
  gnn_w2v_vocab_huffman(vocab);
  return vocab;
}

void
gnn_w2v_skipgram(const char*            text_path,
                 gnn_w2v_vocab_t*       vocab,
                 uint                   sample,
                 uint                   window)
{
  uint i = 0;
  uint a, b, c, d;
  long long l1, l2, target, label;
  uint sentence_length = 0;
  uint sentence_position = 0;
  uint local_iter = 100;
  uint word_index;
  uint negative = 3;
  uint last_word;
  ullong next_random;
  ullong train_words = 0;
  ullong sen[GANN_W2V_MAX_SENTENCE_LENGTH];
  real f, g;
  int hierarchical_softmax = 1;
  real learning_rate = 0.003;

  uint feature_size = 300;
  gnn_w2v_t* w2v = gnn_w2v_new(vocab, feature_size);

  // Allocate the table, 1000 floats.
  real* expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));

    // For each position in the table...
  for (i = 0; i < EXP_TABLE_SIZE; i++)
  {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    expTable[i] = expTable[i] / (expTable[i] + 1);
  }

  for (i = 0; i < vocab->size; i++)
    train_words += vocab->words[i].count;

  FILE *fi = fopen(text_path, "rb");
  fseek(fi, 0, SEEK_SET);

  i = 0;
  char word[128] = {'\0'};
  while(1)
  {
    if (sentence_length == 0)
    {
      while(1)
      {
        gnn_w2v_word_read(word, fi);

        word_index = gnn_w2v_word_index(vocab, word);
//        printf("%d = %s\n", word_index, word);

        if (feof(fi)) break;
        if (word_index == -1) continue;
//        if (word_index == 0 /* skip '<s/>' */) break;

        if (sample > 0)
        {
          real ran = (sqrt(vocab->words[word_index].count / (sample * vocab->size)) + 1) * (sample * train_words) / vocab->words[word_index].count;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word_index;
        sentence_length++;
        if (sentence_length >= GANN_W2V_MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi))
    {
      local_iter--;
      if (local_iter == 0)
      {
        break;
      }
//      sentence_length = 0;
      fseek(fi, 0, SEEK_SET);
//      continue;
    }

    word_index = sen[sentence_position];

    if (word_index == -1) continue;

    for (c = 0; c < feature_size; c++) w2v->hidden_neurons[c] = 0;
    for (c = 0; c < feature_size; c++) w2v->hidden_neurons[c] = 0;

    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;

    /*!
    ** sliding window algorithm
    */
    for (a = b; a < window * 2 + 1 - b; a++)
    {
      if (a != window)
      {

        // Convert the window offset 'a' into an index 'c' into the sentence
        // array.
        c = sentence_position - window + a;
        // Verify c isn't outside the bounds of the sentence.
        if (c < 0) continue;
        if (c >= sentence_length) continue;

        // Get the context word. That is, get the id of the word (its index in
        // the vocab table).
        last_word = sen[c];
//        printf("last word = %d\n", last_word);
        // At this point we have two words identified:
        //   'word' - The word at our current position in the sentence (in the
        //            center of a context window).
        //   'last_word' - The word at a position within the context window.

        // Verify that the word exists in the vocab (I don't think this should
        // ever be the case?)
        if (last_word == -1) continue;
        if (word_index >= vocab->size) continue;
//        printf("(%d, %d)\n", word_index, last_word);
//        printf("(%s, %s)\n", vocab->words[word_index].word, vocab->words[last_word].word);

        // Calculate the index of the start of the weights for 'last_word'.
        l1 = last_word * feature_size;

        for (c = 0; c < feature_size; c++) w2v->hidden_neurons[c] = 0;

        // HIERARCHICAL SOFTMAX
        if (hierarchical_softmax)
        {
          for (d = 0; d < vocab->words[word_index].codelen; d++)
          {
            f = 0;
            l2 = vocab->words[word_index].point[d] * feature_size;
            // Propagate hidden -> output
            for (c = 0; c < feature_size; c++)
              f += w2v->hidden_weights[c + l1] * w2v->output_weights[c + l2];
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            g = (1 - vocab->words[word_index].code[d] - f) * learning_rate;
            // Propagate errors output -> hidden
            for (c = 0; c < feature_size; c++)
              w2v->hidden_neurons[c] += g * w2v->output_weights[c + l2];
            // Learn weights hidden -> output
            for (c = 0; c < feature_size; c++)
              w2v->output_weights[c + l2] += g * w2v->hidden_weights[c + l1];
          }
        }

        // NEGATIVE SAMPLING
        // Rather than performing backpropagation for every word in our
        // vocabulary, we only perform it for a few words (the number of words
        // is given by 'negative').
        // These words are selected using a "unigram" distribution, which is generated
        // in the function InitUnigramTable
        if (negative > 0)
        {
          for (d = 0; d < negative + 1; d++)
          {
            // On the first iteration, we're going to train the positive sample.
            if (d == 0)
            {
              target = word_index;
              label = 1;
            // On the other iterations, we'll train the negative samples.
            } else {
              // Pick a random word to use as a 'negative sample'; do this using
              // the unigram table.

              // Get a random integer.
              next_random = next_random * (unsigned long long)25214903917 + 11;

              // 'target' becomes the index of the word in the vocab to use as
              // the negative sample.
              target = vocab->unigram[(next_random >> 16) % table_size];

              // If the target is the special end of sentence token, then just
              // pick a random word from the vocabulary instead.
              if (target == 0) target = next_random % (vocab->size - 1) + 1;

              // Don't use the positive sample as a negative sample!
              if (target == word_index) continue;

              // Mark this as a negative example.
              label = 0;
            }

            // Get the index of the target word in the output layer.
            l2 = target * feature_size;
            printf("l1 = %lld, l2 = %lld\n", l1, l2);
            /*!
            ** At this point, our two words are represented by their index into
            ** the layer weights.
            ** l1 - The index of our input word within the hidden layer weights.
            ** l2 - The index of our output word within the output layer weights.
            ** label - Whether this is a positive (1) or negative (0) example.
            **
            ** Calculate the dot-product between the input words weights (in
            ** syn0) and the output word's weights (in syn1neg).
            ** Note that this calculates the dot-product manually using a for
            ** loop over the vector elements!
            */
            f = 0;
            for (c = 0; c < feature_size; c++)
              f += w2v->hidden_weights[c + l1] * w2v->negative_samplings[c + l2];

            // This block does two things:
            //   1. Calculates the output of the network for this training
            //      pair, using the expTable to evaluate the output layer
            //      activation function.
            //   2. Calculate the error at the output, stored in 'g', by
            //      subtracting the network output from the desired output,
            //      and finally multiply this by the learning rate.
            if (f > MAX_EXP) g = (label - 1) * learning_rate;
            else if (f < -MAX_EXP) g = (label - 0) * learning_rate;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * learning_rate;

            // Multiply the error by the output layer weights.
            // Accumulate these gradients over the negative samples and the one
            // positive sample.
            for (c = 0; c < feature_size; c++)
              w2v->hidden_neurons[c] += g * w2v->negative_samplings[c + l2];

            // Update the output layer weights by multiplying the output error
            // by the hidden layer weights.
            for (c = 0; c < feature_size; c++)
              w2v->hidden_weights[c + l2] += g * w2v->hidden_weights[c + l1];
          }
        } // if (negative > 0)

        /*!
        ** Once the hidden layer gradients for the negative samples plus the
        ** one positive sample have been accumulated, update the hidden layer
        ** weights.
        ** Note that we do not average the gradient before applying it.
        */
        for (c = 0; c < feature_size; c++) w2v->hidden_weights[c + l1] += w2v->hidden_neurons[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length)
    {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  gnn_w2v_free(w2v);
}

void
gnn_w2v_train(gnn_w2v_vocab_t*          vocab,
              uint                      dims,
              uint                      layer_size,
              const char*               model_file_path)
{
  long a, b, c, d, charv_id;
  long long tot;
  wchar_t ch[10];
  char buf[10], pos;
  float* vec = calloc(dims, sizeof(float));
  FILE *fo;
//  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
//  printf("Starting training using file %s\n", train_file);
//  starting_alpha = alpha;
//  LearnVocabFromTrainFile();
//  if (output_word[0] == 0) return;
//  InitNet();
//  if (negative > 0) InitUnigramTable();
//  start = clock();
//  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
//  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(model_file_path, "wb");
  // Save the word vectors
  fprintf(fo, "%lld\t%d\n", vocab->size, dims);
  for (a = 0; a < vocab->size; a++)
  {
    for (b = 0; vocab->words[a].word[b] != 0; b++)
      fputc(vocab->words[a].word[b], fo);
//    for (b = 0; b < layer_size; b++)
//      fwrite(&syn0[a * layer_size + b], sizeof(float), 1, fo);
//    fputc('\t', fo);
//    for (b = 0; b < dims; b++) vec[b] = 0;
//    for (b = 0; b < dims; b++) vec[b] = syn0[b + a * dim];
//    if (cwe_type && vocab[a].character_size) {
//      for (b = 0; b < vocab[a].character_size; b++) {
//        charv_id = vocab[a].character[b];
//        if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
//          charv_id = get_res_emb(a, b, charv_id);
//        }
//        for (c = 0; c < dims; c++) vec[c] += charv[c + charv_id * dim] / vocab[a].character_size;
//      }
//    }
    for (b = 0; b < dims; b++)
      fprintf(fo, "%lf\t", vec[b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
//  if (strlen(output_char))
//  {
//    fo = fopen(output_char, "wb");
//    if (cwe_type == 5) {
//      tot = 0;
//      for (a = 0; a <= GANN_W2V_MAX_CHINESE - GANN_W2V_MIN_CHINESE; a++)
//        tot += embed_count[a];
//    } else
//      tot = character_size;
//    fprintf(fo, "%lld\t%lld\n", tot, dim);
//    for (a = 0; a < character_size; a++) {
//      if (cwe_type == 1) {
//        ch[0] = GANN_W2V_MIN_CHINESE + a;
//        ch[1] = 0;
//        fprintf(fo, "%ls\ta\t", ch);
//      } else if (cwe_type == 2 || cwe_type == 4) {
//        if (cwe_type == 2)
//          ch[0] = GANN_W2V_MIN_CHINESE + a / 4;
//        else
//          ch[0] = GANN_W2V_MIN_CHINESE + a / 4 / multi_emb;
//        ch[1] = 0;
//        if (cwe_type == 2)
//          c = a % 4;
//        else
//          c = a / multi_emb % 4;
//        if (c == 0)
//          pos = 's';
//        else if (c == 1)
//          pos = 'b';
//        else if (c == 2)
//          pos = 'e';
//        else
//          pos = 'm';
//        fprintf(fo, "%ls\t%c\t", ch, pos);
//      } else if (cwe_type == 3 || cwe_type == 5) {
//        ch[0] = GANN_W2V_MIN_CHINESE + a / multi_emb;
//        ch[1] = 0;
//        c = a % multi_emb;
//        if (cwe_type == 5 && c >= embed_count[a / multi_emb])
//          continue;
//        fprintf(fo, "%ls\ta\t", ch);
//      }
//      for (b = 0; b < dims; b++)
//        fprintf(fo, "%lf\t", charv[b + dims * a]);
//      fprintf(fo, "\n");
//    }
//    fclose(fo);
//  }
}


gnn_w2v_t*
gnn_w2v_new(gnn_w2v_vocab_t* vocab, uint dimensions)
{
  gnn_w2v_t* ret = (gnn_w2v_t*) malloc(sizeof(gnn_w2v_t));
  ret->vocab_size = vocab->size;
  ret->dim_num = dimensions;
  ret->char_size = vocab->char_size;

  int rs = 0;
  int i, j;
  ullong next_random = 1;

  rs = posix_memalign((void **)&ret->hidden_weights,
                      128,
                      ret->vocab_size * ret->dim_num * sizeof(real));
  if (ret->hidden_weights == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for hidden weights in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (i = 0; i < ret->vocab_size; i++)
    for (j = 0; j < ret->dim_num; j++)
    {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      ret->hidden_weights[i * ret->dim_num + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / ret->dim_num;
    }

  rs = posix_memalign((void **)&ret->output_weights,
                      128,
                      ret->vocab_size * ret->dim_num * sizeof(real));
  if (ret->output_weights == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for output weights in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (i = 0; i < ret->vocab_size; i++)
    for (j = 0; j < ret->dim_num; j++)
    {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      ret->output_weights[i * ret->dim_num + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / ret->dim_num;
    }

  rs = posix_memalign((void **)&ret->negative_samplings,
                      128,
                      ret->vocab_size * ret->dim_num * sizeof(real));
  if (ret->negative_samplings == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for negative samplings in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (i = 0; i < ret->vocab_size; i++)
    for (j = 0; j < ret->dim_num; j++)
    {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      ret->negative_samplings[i * ret->dim_num + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / ret->dim_num;
    }

  rs = posix_memalign((void **)&ret->hidden_neurons,
                      128,
                      ret->dim_num * sizeof(real));
  if (ret->hidden_neurons == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for hidden neurons in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (j = 0; j < ret->dim_num; j++)
    ret->hidden_neurons[j] = 0;

  rs = posix_memalign((void **)&ret->softmax_neurons,
                      128,
                      ret->dim_num * sizeof(real));
  if (ret->softmax_neurons == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for softmax neurons in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (j = 0; j < ret->dim_num; j++)
    ret->softmax_neurons[j] = 0;

  rs = posix_memalign((void **)&ret->negative_samplings,
                      128,
                      ret->vocab_size * ret->dim_num * sizeof(real));
  if (ret->negative_samplings == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for negative samplings in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }

  rs = posix_memalign((void **)&ret->char_weights,
                      128,
                      ret->char_size * ret->dim_num * sizeof(real));
  if (ret->char_weights == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for character weights in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (i = 0; i < (long long)ret->char_size * ret->dim_num; i++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    ret->char_weights[i] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / ret->dim_num;
  }

  rs = posix_memalign((void **)&ret->embedded_count,
                      128,
                      (GANN_W2V_MAX_CHINESE - GANN_W2V_MIN_CHINESE) * sizeof(uint));
  if (ret->embedded_count == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for embedded count in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }
  for (i = 0; i < (GANN_W2V_MAX_CHINESE - GANN_W2V_MIN_CHINESE + 1); i++) {
    ret->embedded_count[i] = 0;
  }

  rs = posix_memalign((void **)&ret->last_embedded_count,
                      128,
                      (GANN_W2V_MAX_CHINESE - GANN_W2V_MIN_CHINESE) * sizeof(uint));
  if (ret->last_embedded_count == NULL)
  {
    fprintf(stderr, "error: failed to allocate memories for last embedded count in %d of %s.\n", __LINE__, __FILE__);
    exit(1);
  }

  return ret;
}

void
gnn_w2v_free(gnn_w2v_t* w2v)
{
  if (w2v->hidden_neurons != NULL)
    free(w2v->hidden_neurons);
  if (w2v->hidden_weights != NULL)
    free(w2v->hidden_weights);
  if (w2v->output_weights != NULL)
    free(w2v->output_weights);
  if (w2v->softmax_neurons != NULL)
    free(w2v->softmax_neurons);
  if (w2v->char_weights != NULL)
    free(w2v->char_weights);
  if (w2v->embedded_count != NULL)
    free(w2v->embedded_count);
  if (w2v->negative_samplings != NULL)
    free(w2v->negative_samplings);
  free(w2v);
}
