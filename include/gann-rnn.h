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

struct gnn_rnn_s
{

  /*!
  ** weights input to hidden
  */
  float* wxh;

  /*!
  ** weights hidden to hidden
  */
  float* whh;

  /*!
  ** weights hidden to output
  */
  float* why;

  /*!
  ** weights previous hidden state
  */
  float* hprev;

  /*!
  ** weights of hidden state for sampling
  */
  float* hpredict;

  /*!
  ** temporary matrix to store weights of hidden state for sampling
  */
  float* hptemp;

  /*!
  ** input state
  */
  float* xs;

  /*!
  ** current hidden state
  */
  float* hs;

  /*!
  ** output state
  */
  float* ys;

  /*!
  ** output probabilities
  */
  float* ps;

  /*!
  ** change in weight input to hidden
  */
  float* delwxh;

  /*!
  ** change in weight hidden to hidden
  */
  float* delwhh;

  /*!
  ** change in weight hidden to output
  */
  float* delwhy;

  /*!
  ** delta current hidden to next
  */
  float* dhnext;

  /*!
  ** output gradient
  */
  float* dy;

  /*!
  ** hidden gradient
  */
  float* dh;

  /*!
  ** hidden gradient
  */
  float* dhraw;

  /*!
  ** memory variable for adagrad
  */
  float* mWxh;

  /*!
  ** memory variable for adagrad
  */
  float* mWhh;

  /*!
  ** memory variable for adagrad
  */
  float* mWhy;

  /*!
  ** cumulative sum for non-uniform probability distribution
  */
  float* cdf;

};

#ifdef __cplusplus
}
#endif

#endif // __GNN_MLP_H__
