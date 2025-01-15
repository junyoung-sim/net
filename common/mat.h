#ifndef __MAT_H_
#define __MAT_H_

//==========================================================
// Vector (1D)
//==========================================================

typedef struct Vec Vec;
struct Vec
{
  int    size;
  float* dat;
};

Vec* make_vec
(
  int   size,
  float init
);

void dump_vec
(
  Vec* vec
);

void free_vec
(
  Vec* vec
);

//==========================================================
// Matrix (2D)
//==========================================================

typedef struct Mat Mat;
struct Mat
{
  int     size[2];
  float** dat;
};

Mat* make_mat
(
  int   rows,
  int   cols,
  float init
);

void dump_mat
(
  Mat* mat
);

void free_mat
(
  Mat* mat
);

//==========================================================
// Operations
//==========================================================

void mat_vec_product
(
  Mat* mat,
  Vec* vec,
  Vec* out
);

void vec_sum
(
  Vec* vec, 
  Vec* diff
);

#endif