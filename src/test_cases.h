// Reference file 
// (for copying test values into C file(s) 
// and knowing what to reference in Python files.)

#ifndef TEST_CASES_H
#define TEST_CASES_H

// 2D Matrix test cases
int test_values1[3][3] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

int test_values2[3][3] = {
    {9, 8, 7},
    {6, 5, 4},
    {3, 2, 1}
};

// Vector
float test_float_values7[] = {1, 2, 3};
float test_float_values8[] = {4, 5, 6};
int test_shape_vec[1] = {3};

// 3D Tensor
int test_shape_3d[3] = {2, 3, 4};
float test_values_3d[24] = {
    1, 2, 3, 4,    5, 6, 7, 8,    9, 10, 11, 12,
    13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24
};

// 4D Tensor
int test_shape_4d[4] = {2, 2, 2, 3};
float test_values_4d[24] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
};

// Basic shapes
int test_shape_2d[2] = {3, 3};

// Scalar values
float test_3d_scalar = 2.0f;

// Batched matrix multiplication
int test_batch_shape_a[3] = {2, 2, 3};
float test_batch_a_vals[12] = {1, 2, 3, 4, 5, 6,    7, 8, 9, 10, 11, 12};

int test_batch_shape_b[3] = {2, 3, 2};
float test_batch_b_vals[12] = {1, 2, 3, 4, 5, 6,    7, 8, 9, 10, 11, 12};

// Test parameters
float test_scalar_multiplier = 3.1f;

#endif
