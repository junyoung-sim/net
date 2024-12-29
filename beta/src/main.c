#include <stdio.h>
#include <stdlib.h>

#include "../common/mat.h"

int main(int argc, char *argv[])
{
  Vec* vec = make_vec(5, 1.00f);
  dump_vec(vec);
  free_vec(vec);

  Mat* mat = make_mat(3, 3, 1.00f);
  dump_mat(mat);
  free_mat(mat);

  return 0;
}