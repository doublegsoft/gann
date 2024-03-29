#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "gann.h"
#include "gann-mlp.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

const char *iris_data = "../../data/iris.data";

float *input, *class;
int samples;
const char *class_names[] =
    { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

void load_data()
{
  /* Load the iris data-set. */
  FILE *in = fopen(iris_data, "r");
  if (!in)
  {
    printf("Could not open file: %s\n", iris_data);
    exit(1);
  }

  /* Loop through the data to get a count. */
  char line[1024];
  while (!feof(in) && fgets(line, 1024, in))
  {
    ++samples;
  }
  fseek(in, 0, SEEK_SET);

  printf("Loading %d data points from %s\n", samples, iris_data);

  /*!
  ** allocate memory for input and output data.
  */
  input = malloc(sizeof(float) * samples * 4);
  class = malloc(sizeof(float) * samples * 3);

  /* Read the file into our arrays. */
  int i, j;
  for (i = 0; i < samples; ++i)
  {
    float *p = input + i * 4;
    float *c = class + i * 3;
    c[0] = c[1] = c[2] = 0.0;

    if (fgets(line, 1024, in) == NULL)
    {
      perror("fgets");
      exit(1);
    }

    char *split = strtok(line, ",");
    for (j = 0; j < 4; ++j)
    {
      p[j] = atof(split);
      split = strtok(0, ",");
    }

    split[strlen(split) - 1] = 0;
    if (strcmp(split, class_names[0]) == 0)
    {
      c[0] = 1.0;
    } else if (strcmp(split, class_names[1]) == 0)
    {
      c[1] = 1.0;
    } else if (strcmp(split, class_names[2]) == 0)
    {
      c[2] = 1.0;
    } else
    {
      printf("Unknown class %s.\n", split);
      exit(1);
    }
//    printf("Data point %d is %f %f %f %f  ->   %f %f %f\n", i, p[0], p[1], p[2], p[3], c[0], c[1], c[2]);
  }

  fclose(in);
}

int main(int argc, char *argv[])
{
  printf("Train an ANN on the IRIS dataset using backpropagation.\n");

  srand(time(0));

  /* Load the data from file. */
  load_data();

  /* 4 inputs.
   * 2 hidden layer(s) of 6 neurons.
   * 3 outputs (1 per class)
   */
  gnn_mlp_t* mlp = gnn_mlp_new(4, 2, 6, 3);

  int i, j;
  int loops = 5000;

  /* Train the network with backpropagation. */

  for (i = 0; i < loops; ++i)
  {
    for (j = 0; j < samples; ++j)
    {
      gnn_mlp_train(mlp, input + j * 4, class + j * 3, .01);
    }
  }
  printf("trained for %d loops over data.\n", loops);

  int correct = 0;
  for (j = 0; j < samples; ++j)
  {
    const float *guess = gnn_mlp_forward(mlp, input + j * 4);
    if (class[j * 3 + 0] == 1.0)
    {
      if (guess[0] > guess[1] && guess[0] > guess[2])
        ++correct;
    }
    else if (class[j * 3 + 1] == 1.0)
    {
      if (guess[1] > guess[0] && guess[1] > guess[2])
        ++correct;
    }
    else if (class[j * 3 + 2] == 1.0)
    {
      if (guess[2] > guess[0] && guess[2] > guess[1])
        ++correct;
    }
    else
    {
      printf("Logic error.\n");
       exit(1);
    }
    gnn_vec_print(guess, 3);
  }

  printf("%d/%d correct (%0.1f%%).\n", correct, samples,
      (float) correct / samples * 100.0);

  gnn_mlp_free(mlp);
  free(input);
  free(class);

  FILE* iris = fopen("./iris-model.txt", "w");
  gnn_mlp_write(mlp, iris);
  fclose(iris);

  return 0;
}
