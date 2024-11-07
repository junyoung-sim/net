#ifndef __MAT_H_
#define __MAT_H_

typedef struct Vec Vec;
struct Vec
{
    int size;
    float *dat;
};

typedef struct Mat Mat;
struct Mat
{
    int size[2];
    float **dat;
};

Vec *make_vec(int size, float init);
Mat *make_mat(int rows, int cols, float init);

void mat_vec_product(Mat *mat, Vec *vec, Vec *out);
void vec_sum(Vec *vec, Vec *diff);

void copy_vec(Vec *src, Vec *dst);

void dump_vec(Vec *vec);
void dump_mat(Mat *mat);

void free_vec(Vec *vec);
void free_mat(Mat *mat);

#endif