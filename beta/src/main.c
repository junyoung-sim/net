#include <stdio.h>
#include <stdlib.h>

#include "../common/mat.h"

int main(int argc, char *argv[])
{
  Vec* vec = make_vec(3, 1.00f);
  dump_vec(vec);

  Mat* mat = make_mat(5, 3, 1.00f);
  dump_mat(mat);

  Vec* prod = make_vec(5, 0.00f);
  mat_vec_product(mat, vec, prod);
  dump_vec(prod);

  free_vec(vec);
  free_mat(mat);
  free_vec(prod);

  return 0;
}