#include <assert.h>
#include <gfc.h>

#include "gann-w2v.h"

int
main(int argc, char* argv[])
{
  int i, j;
  gnn_w2v_vocab_t* vocab = gnn_w2v_read("../../data/analogy.txt");
  FILE* out = fopen("../../debug.txt", "w");

  for (i = 0; i < vocab->size; i++)
  {
    char initial[5] = {'\0'};
    gnn_w2v_word_t* word = &vocab->words[i];
    int len = gfc_utf8_initial(word->word, initial);
    fprintf(out, "%s = %d, initial = %s, bytes = ", word->word, gfc_utf8_length(word->word), initial);
    for (j = 0; j < len; j++)
      fprintf(out, "%02X", (unsigned char)initial[j]);
    fprintf(out, ", count = %lld, code = ", word->count);
    for (j = 0; j < word->codelen; j++)
      fprintf(out, "%s", word->code[j] == 0 ? "0" : "1");
    fprintf(out, ", codelen = %d", (int)word->codelen);
    fprintf(out, "\n");
  }
  assert(vocab != NULL);
  gnn_w2v_train(vocab, 100, 100, "./analogy.bin");

  gnn_w2v_t* w2v = gnn_w2v_build(vocab, 100);
  return 0;
}
