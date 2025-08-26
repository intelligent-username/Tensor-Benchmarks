#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "test_cases.h"
#include <math.h>

typedef struct Tensor {
    int ndim;           // # of dimensions
    int *shape;         // [dim0, dim1, ..., dimN]
    int *strides;       // For indexing
    int size;           // Total number of elements
    float *data;        // Flattened
} Tensor;

/*
 * @brief Calculate strides for a tensor based on its shape.
 *
 * @param tensor Pointer to the Tensor structure.
 */
void calculate_strides(Tensor *tensor) {
    tensor->strides = malloc(tensor->ndim * sizeof *tensor->strides);
    tensor->strides[tensor->ndim - 1] = 1;
    for (int i = tensor->ndim - 2; i >= 0; i--) {
        tensor->strides[i] = tensor->strides[i + 1] * tensor->shape[i + 1];
    }
}

/*
 * @brief Get the flat index in the data array from multi-dimensional indices.
 *
 * @param tensor Pointer to the Tensor structure.
 * @param indices Array of indices for each dimension.
 * @return The flat index in the data array.
 */
int get_flat_index(Tensor *tensor, int *indices) {
    int flat_idx = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        flat_idx += indices[i] * tensor->strides[i];
    }
    return flat_idx;
}

/*
 * @brief Check if two tensors have the same shape.
 *
 * @param t1 Pointer to the first Tensor structure.
 * @param t2 Pointer to the second Tensor structure.
 * @return 1 if shapes are equal, 0 otherwise.
 */
int shapes_equal(Tensor *t1, Tensor *t2) {
    if (t1->ndim != t2->ndim) return 0;
    for (int i = 0; i < t1->ndim; i++) {
        if (t1->shape[i] != t2->shape[i]) return 0;
    }
    return 1;
}

/*
 * @brief Initialize a tensor with the given shape and values.
 *
 * @param ndim Number of dimensions.
 * @param shape Array representing the shape of the tensor.
 * @param values Array of values to initialize the tensor (can be NULL).
 * @return The initialized Tensor structure.
 */
Tensor Tensor_Initializer(int ndim, int *shape, float *values) {
    Tensor self;
    self.ndim = ndim;
    self.shape = malloc(ndim * sizeof *self.shape);
    memcpy(self.shape, shape, ndim * sizeof(int));
    
    // Calculate total size
    self.size = 1;
    for (int i = 0; i < ndim; i++) {
        self.size *= shape[i];
    }
    
    self.data = malloc(self.size * sizeof *self.data);
    calculate_strides(&self);
    
    // Initialize data
    for (int i = 0; i < self.size; i++) {
        if (values) {
            self.data[i] = values[i];
        } else {
            self.data[i] = 0;
        }
    }
    
    return self;
}

/*
 * @brief Perform matrix multiplication for tensors.
 *
 * @param t1 The first Tensor operand.
 * @param t2 The second Tensor operand.
 * @return The resulting Tensor after matrix multiplication.
 */
Tensor Tensor_Multiplier(Tensor t1, Tensor t2) {
    // Matrix multiplication for tensors (operates on last two dimensions)
    // Assumes both tensors have at least 2 dimensions
    
    if (t1.ndim < 2 || t2.ndim < 2) {
        printf("Error: Both tensors must have at least 2 dimensions for matrix multiplication\n");
        Tensor empty = {0};
        return empty;
    }
    
    // Get matrix dimensions (last two dimensions)
    int t1_rows = t1.shape[t1.ndim - 2];
    int t1_cols = t1.shape[t1.ndim - 1];
    int t2_rows = t2.shape[t2.ndim - 2];
    int t2_cols = t2.shape[t2.ndim - 1];
    
    if (t1_cols != t2_rows) {
        printf("Error: Matrix dimensions incompatible (%d != %d)\n", t1_cols, t2_rows);
        Tensor empty = {0};
        return empty;
    }
    
    // Calculate output shape (batch dims + matrix result dims)
    int max_batch_dims = (t1.ndim - 2) > (t2.ndim - 2) ? (t1.ndim - 2) : (t2.ndim - 2);
    int output_ndim = max_batch_dims + 2;
    int *output_shape = malloc(output_ndim * sizeof *output_shape);
    
    // Handle batch dimensions (simplified - assumes compatible batch dimensions)
    for (int i = 0; i < max_batch_dims; i++) {
        int t1_dim = i < (t1.ndim - 2) ? t1.shape[i] : 1;
        int t2_dim = i < (t2.ndim - 2) ? t2.shape[i] : 1;
        output_shape[i] = (t1_dim > t2_dim) ? t1_dim : t2_dim;
    }
    
    // Set matrix result dimensions
    output_shape[output_ndim - 2] = t1_rows;
    output_shape[output_ndim - 1] = t2_cols;
    
    Tensor result = Tensor_Initializer(output_ndim, output_shape, NULL);
    
    // Calculate number of matrix multiplications to perform
    int batch_size = 1;
    for (int i = 0; i < max_batch_dims; i++) {
        batch_size *= output_shape[i];
    }
    
    // Perform batched matrix multiplication
    // Reuse a stack buffer for batch indices and compute flat offsets via strides
    for (int batch = 0; batch < batch_size; batch++) {
    int batch_indices[32]; // Use a fixed-size array, assuming max_batch_dims <= 32
        int temp_batch = batch;
        for (int bi = max_batch_dims - 1; bi >= 0; bi--) {
            batch_indices[bi] = temp_batch % output_shape[bi];
            temp_batch /= output_shape[bi];
        }

        // Compute batch base offsets for t1, t2, and result
        int t1_batch_base = 0;
        int t2_batch_base = 0;
        int res_batch_base = 0;
        for (int d = 0; d < max_batch_dims; d++) {
            int idx = batch_indices[d];
            if (d < t1.ndim - 2) t1_batch_base += idx * t1.strides[d];
            if (d < t2.ndim - 2) t2_batch_base += idx * t2.strides[d];
            res_batch_base += idx * result.strides[d];
        }

        int t1_row_stride = t1.strides[t1.ndim - 2];
        int t1_col_stride = t1.strides[t1.ndim - 1];
        int t2_row_stride = t2.strides[t2.ndim - 2];
        int t2_col_stride = t2.strides[t2.ndim - 1];
        int res_row_stride = result.strides[result.ndim - 2];
        int res_col_stride = result.strides[result.ndim - 1];

        for (int i = 0; i < t1_rows; i++) {
            for (int j = 0; j < t2_cols; j++) {
                float sum = 0;
                for (int k = 0; k < t1_cols; k++) {
                    int t1_idx = t1_batch_base + i * t1_row_stride + k * t1_col_stride;
                    int t2_idx = t2_batch_base + k * t2_row_stride + j * t2_col_stride;
                    sum += t1.data[t1_idx] * t2.data[t2_idx];
                }
                int res_idx = res_batch_base + i * res_row_stride + j * res_col_stride;
                result.data[res_idx] = sum;
            }
        }
    }
    
    free(output_shape);
    return result;
}

/*
 * @brief Perform element-wise addition of two tensors.
 *
 * @param t1 The first Tensor operand.
 * @param t2 The second Tensor operand.
 * @return The resulting Tensor after addition.
 */
Tensor Tensor_Adder(Tensor t1, Tensor t2) {
    // Element-wise addition - tensors must have same shape
    if (!shapes_equal(&t1, &t2)) {
        printf("Error: Tensors must have the same shape for addition\n");
        Tensor empty = {0};
        return empty;
    }
    
    Tensor result = Tensor_Initializer(t1.ndim, t1.shape, NULL);
    
    for (int i = 0; i < t1.size; i++) {
        result.data[i] = t1.data[i] + t2.data[i];
    }
    
    return result;
}

/*
 * @brief Multiply a tensor by a scalar value.
 *
 * @param t The Tensor operand.
 * @param c The scalar value.
 * @return The resulting Tensor after scalar multiplication.
 */
Tensor Scalar_Multiplication(Tensor t, float c) {
    Tensor result = Tensor_Initializer(t.ndim, t.shape, NULL);
    
    for (int i = 0; i < t.size; i++) {
        result.data[i] = t.data[i] * c;
    }
    
    return result;
}

/*
 * @brief Compute the dot product of two vectors.
 *
 * @param t1 The first Tensor operand (1D tensor).
 * @param t2 The second Tensor operand (1D tensor).
 * @return The resulting scalar value of the dot product.
 */
float Dot_Product(Tensor t1, Tensor t2) {
    // Dot product (vectors only)x
    if (t1.ndim != 1 || t2.ndim != 1) {
        printf("Error: Dot product only works for 1D tensors (vectors)\n");
        return 0;
    }
    
    if (t1.shape[0] != t2.shape[0]) {
        printf("Error: Vectors must have the same length for dot product\n");
        return 0;
    }
    
    float sum = 0;
    for (int i = 0; i < t1.shape[0]; i++) {
        sum += t1.data[i] * t2.data[i];
    }
    
    return sum;
}

/*
 * @brief Free the memory allocated for a tensor.
 *
 * @param t The Tensor structure to free.
 */
void Tensor_Freer(Tensor t) {
    if (t.data) free(t.data);
    if (t.shape) free(t.shape);
    if (t.strides) free(t.strides);
}

/*
 * @brief Print the contents of a tensor in a human-readable format.
 *
 * @param t The Tensor structure to print.
 */
void Tensor_Printer(Tensor t) {
    // Print shape
    printf("shape=%d", t.shape[0]);
    for (int i = 1; i < t.ndim; i++) {
        printf(",%d", t.shape[i]);
    }
    printf("\n");

    // Recursive pretty print for n-dim tensors
    print_recursive(t.data, t.shape, t.ndim, 0, 0);
    printf("\n\n");
}

// Move print_recursive outside Tensor_Printer
void print_recursive(float *data, int *shape, int ndim, int depth, int offset) {
    if (depth == ndim - 1) {
        printf("[");
        for (int i = 0; i < shape[depth]; i++) {
            printf("%g", data[offset + i]);
            if (i < shape[depth] - 1) printf(", ");
        }
        printf("]");
    } else {
        printf("[");
        int stride = 1;
        for (int i = depth + 1; i < ndim; i++) stride *= shape[i];
        for (int i = 0; i < shape[depth]; i++) {
            print_recursive(data, shape, ndim, depth + 1, offset + i * stride);
            if (i < shape[depth] - 1) printf(",\n ");
        }
        printf("]");
    }
}


/*
 * @brief Get the current time in seconds.
 *
 * @return The current time in seconds as a double.
 */
static double now_seconds() {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

typedef struct BenchInput {
    char op[8];
    Tensor A;
    Tensor B;
    int has_B;
    float scalar;
    int has_scalar;
} BenchInput;

/*
 * @brief Read the next token from a file.
 *
 * @param f Pointer to the file.
 * @param buf Buffer to store the token.
 * @param n Size of the buffer.
 * @return 1 if a token was read, 0 otherwise.
 */
static int read_token(FILE *f, char *buf, size_t n) {
    int c;
    // skip whitespace
    do {
        c = fgetc(f);
        if (c == EOF) return 0;
    } while (c==' '||c=='\n'||c=='\t' || c=='\r');
    size_t i = 0;
    buf[i++] = (char)c;
    while (i < n-1) {
        c = fgetc(f);
        if (c==EOF || c==' '||c=='\n'||c=='\t'||c=='\r') break;
        buf[i++] = (char)c;
    }
    buf[i] = '\0';
    return 1;
}

/*
 * @brief Parse an integer from a file.
 *
 * @param f Pointer to the file.
 * @return The parsed integer.
 */
static int parse_int(FILE *f) {
    char buf[128];  // Increased buffer size
    if (!read_token(f, buf, sizeof buf)) return 0;
    return atoi(buf);
}

/*
 * @brief Parse a floating-point number from a file.
 *
 * @param f Pointer to the file.
 * @return The parsed float.
 */
static float parse_float(FILE *f) {
    char buf[128];  // Increased buffer size
    if (!read_token(f, buf, sizeof buf)) return 0.0f;
    return (float)atof(buf);
}

/*
 * @brief Parse a tensor from a file.
 *
 * @param f Pointer to the file.
 * @return The parsed Tensor structure.
 */
static Tensor parse_tensor(FILE *f) {
    // expecting: <TAG> SHAPE d dims DATA <v... or scalar>
    char buf[64];
    // read TAG (A or B)
    if (!read_token(f, buf, sizeof buf)) { Tensor empty={0}; return empty; }
    // SHAPE
    read_token(f, buf, sizeof buf); // SHAPE
    int d = parse_int(f);
    int *shape = malloc(sizeof(int)*d);
    for (int i=0;i<d;i++) shape[i] = parse_int(f);
    // DATA
    read_token(f, buf, sizeof buf); // DATA
    int total = 1; for (int i=0;i<d;i++) total *= shape[i];
    // Peek next token by reading it, then decide if we should treat it as scalar-fill or explicit list
    if (!read_token(f, buf, sizeof buf)) { Tensor empty={0}; free(shape); return empty; }
    // First numeric value
    float first = (float)atof(buf);
    float *vals = malloc(sizeof(float)*total);
    vals[0] = first;
    int count = 1;
    long pos;
    
    // Limit explicit value reading to avoid hanging on huge tensors
    int max_explicit = 1024; // Stop reading explicit values after this many
    while (count < total && count < max_explicit) {
        pos = ftell(f);
        if (!read_token(f, buf, sizeof buf)) break;
        char c0 = buf[0];
        int isnum = (c0=='-'||c0=='+'||c0=='.'||(c0>='0'&&c0<='9'));
        if (!isnum) { fseek(f, pos, SEEK_SET); break; }
        vals[count++] = (float)atof(buf);
    }
    if (count < total) {
        // scalar fill - consume any remaining numeric tokens to align parser
        while (count < max_explicit) {
            pos = ftell(f);
            if (!read_token(f, buf, sizeof buf)) break;
            char c0 = buf[0];
            int isnum = (c0=='-'||c0=='+'||c0=='.'||(c0>='0'&&c0<='9'));
            if (!isnum) { fseek(f, pos, SEEK_SET); break; }
            count++;
        }
        for (int i = (count < max_explicit ? count : 1); i < total; ++i) vals[i] = first;
    }
    Tensor t = Tensor_Initializer(d, shape, vals);
    free(vals);
    free(shape);
    return t;
}

/*
 * @brief Parse a single benchmark case from a file.
 *
 * @param f Pointer to the file.
 * @param out Pointer to the BenchInput structure to store the parsed case.
 * @return 1 if a case was parsed, 0 otherwise.
 */
static int parse_one_case(FILE *f, BenchInput *out) {
    char buf[64];
    // find next OP
    int have = 0;
    while (read_token(f, buf, sizeof buf)) {
        if (strcmp(buf, "OP") == 0) { have = 1; break; }
    }
    if (!have) return 0; // no more cases
    if (!read_token(f, buf, sizeof buf)) return 0; // op name
    strncpy(out->op, buf, sizeof(out->op)-1); out->op[sizeof(out->op)-1]='\0';
    out->has_B = 0; out->has_scalar = 0;
    if (strcmp(out->op, "ADD")==0 || strcmp(out->op, "MM")==0 || strcmp(out->op, "DOT")==0 || strcmp(out->op, "BMM")==0) {
        out->A = parse_tensor(f);
        out->B = parse_tensor(f);
        out->has_B = 1;
    } else if (strcmp(out->op, "SCAL")==0) {
        out->A = parse_tensor(f);
        read_token(f, buf, sizeof buf); // SCALAR
        out->scalar = parse_float(f); out->has_scalar = 1;
    } else {
        return 0;
    }
    return 1;
}

/*
 * @brief Run a benchmark for a given input.
 *
 * @param in Pointer to the BenchInput structure containing the benchmark input.
 * @param runs Number of benchmark runs.
 * @param warmups Number of warmup runs.
 * @return 0 on success.
 */
static int run_bench(const BenchInput *in, int runs, int warmups) {
    // warmups
    for (int w=0; w<warmups; ++w) {
        if (strcmp(in->op, "ADD")==0) {
            Tensor r = Tensor_Adder(in->A, in->B); Tensor_Freer(r);
        } else if (strcmp(in->op, "MM")==0 || strcmp(in->op, "BMM")==0) {
            Tensor r = Tensor_Multiplier(in->A, in->B); Tensor_Freer(r);
        } else if (strcmp(in->op, "SCAL")==0) {
            Tensor r = Scalar_Multiplication(in->A, in->scalar); Tensor_Freer(r);
        } else if (strcmp(in->op, "DOT")==0) {
            volatile float s = Dot_Product(in->A, in->B); (void)s;
        }
    }

    double *times = malloc(sizeof(double)*runs);
    for (int r=0; r<runs; ++r) {
        double t0 = now_seconds();
        if (strcmp(in->op, "ADD")==0) {
            Tensor res = Tensor_Adder(in->A, in->B);
            double t1 = now_seconds(); times[r] = t1 - t0; Tensor_Freer(res);
        } else if (strcmp(in->op, "MM")==0 || strcmp(in->op, "BMM")==0) {
            Tensor res = Tensor_Multiplier(in->A, in->B);
            double t1 = now_seconds(); times[r] = t1 - t0; Tensor_Freer(res);
        } else if (strcmp(in->op, "SCAL")==0) {
            Tensor res = Scalar_Multiplication(in->A, in->scalar);
            double t1 = now_seconds(); times[r] = t1 - t0; Tensor_Freer(res);
        } else if (strcmp(in->op, "DOT")==0) {
            volatile float s = Dot_Product(in->A, in->B);
            double t1 = now_seconds(); times[r] = t1 - t0; (void)s;
        } else {
            times[r] = 0.0;
        }
    }
    // compute stats
    double sum=0.0, min=1e99, max=0.0; for (int i=0;i<runs;i++){sum+=times[i]; if(times[i]<min)min=times[i]; if(times[i]>max)max=times[i];}
    double mean = sum / (runs>0?runs:1);
    // median
    for (int i=0;i<runs;i++){
        for (int j=i+1;j<runs;j++) if (times[j] < times[i]) { double tmp=times[i]; times[i]=times[j]; times[j]=tmp; }
    }
    double median = runs? (runs%2? times[runs/2] : 0.5*(times[runs/2-1]+times[runs/2])) : 0.0;
    // population stdev
    double var=0.0; for (int i=0;i<runs;i++){ double d=times[i]-mean; var += d*d; }
    double stdev = runs? (var / runs) : 0.0; if (stdev>0) stdev = sqrt(stdev);

    // print JSON
    printf("{\"op\":\"%s\",\"shape_A\":[", in->op);
    for (int i=0;i<in->A.ndim;i++){ printf("%d%s", in->A.shape[i], (i<in->A.ndim-1)?",":""); }
    printf("],\"shape_B\":");
    if (in->has_B){ printf("["); for (int i=0;i<in->B.ndim;i++){ printf("%d%s", in->B.shape[i], (i<in->B.ndim-1)?",":""); } printf("]"); }
    else printf("null");
    printf(",\"runs\":%d,\"warmups\":%d,\"median\":%.12g,\"mean\":%.12g,\"stdev\":%.12g,\"min\":%.12g,\"max\":%.12g,\"times\":[",
           runs, warmups, median, mean, stdev, min, max);
    for (int i=0;i<runs;i++) {
        printf("%.12g%s", times[i], (i<runs-1)?",":"");
    }
    printf("]}\n");

    free(times);
    return 0;
}

/*
 * @brief Main function to run benchmarks or test examples.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Exit code.
 */
int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--bench") == 0) {
        const char *path = argv[2];
        int runs = 10, warmups = 3;
        const char *env_runs = getenv("BENCH_RUNS");
        const char *env_warmups = getenv("BENCH_WARMUPS");
        if (env_runs) runs = atoi(env_runs);
        if (env_warmups) warmups = atoi(env_warmups);
        FILE *f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Failed to open bench file: %s\n", path); return 1; }
        int rc = 0;
        for (;;) {
            BenchInput in;
            if (!parse_one_case(f, &in)) break;
            rc |= run_bench(&in, runs, warmups);
            Tensor_Freer(in.A);
            if (in.has_B) Tensor_Freer(in.B);
        }
        fclose(f);
        return rc;
    }

    printf("Testing C version.\n");

    printf("Test examples.\n");

    // Build matrices as 2D tensors directly
    float flat_values1[9], flat_values2[9];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            flat_values1[i * 3 + j] = (float)test_values1[i][j];
            flat_values2[i * 3 + j] = (float)test_values2[i][j];
        }
    }
    Tensor matrix1 = Tensor_Initializer(2, test_shape_2d, flat_values1);
    Tensor matrix2 = Tensor_Initializer(2, test_shape_2d, flat_values2);

    printf("\n=== MATRIX ADDITION TEST ===\n");
    printf("Matrix 1:\n");

    Tensor_Printer(matrix1);
    
    printf("Matrix 2:\n");
    
    Tensor_Printer(matrix2);
    
    Tensor matrix3 = Tensor_Adder(matrix1, matrix2);
    
    printf("Result (Matrix 1 + Matrix 2):\n");
    
    Tensor_Printer(matrix3);

    printf("\n=== MATRIX MULTIPLICATION TEST ===\n");
    printf("Matrix 1:\n");

    Tensor_Printer(matrix1);
    
    printf("Matrix 2:\n");
    
    Tensor_Printer(matrix2);
    Tensor matrix4 = Tensor_Multiplier(matrix1, matrix2);
    
    printf("Result (Matrix 1 * Matrix 2):\n");

    Tensor_Printer(matrix4);

    float c = test_scalar_multiplier;
    
    printf("\n=== SCALAR MULTIPLICATION TEST ===\n");
    printf("Matrix 1:\n");
    
    Tensor_Printer(matrix1);
    
    printf("Scalar: %g\n", c);
    
    Tensor matrix5 = Scalar_Multiplication(matrix1, c);
    
    printf("Result (Matrix 1 * %g):\n", c);
    
    Tensor_Printer(matrix5);

    printf("\n=== VECTOR DOT PRODUCT TEST ===\n");
    
    Tensor vector1 = Tensor_Initializer(1, test_shape_vec, test_float_values7);
    
    Tensor vector2 = Tensor_Initializer(1, test_shape_vec, test_float_values8);
    
    printf("Vector 1:\n");
    
    Tensor_Printer(vector1);
    
    printf("Vector 2:\n");
    
    Tensor_Printer(vector2);
    
    float dot_result = Dot_Product(vector1, vector2);
    
    printf("Result (Dot product): %g\n\n", dot_result);

    printf("\n=== N-DIMENSIONAL TENSOR TESTS ===\n");
    // Test 3D tensor creation and operations
    
    printf("Testing 3D Tensor:\n");
    
    Tensor tensor3d = Tensor_Initializer(3, test_shape_3d, test_values_3d);
    
    printf("Original 3D Tensor (2x3x4):\n");
    
    Tensor_Printer(tensor3d);
    
    printf("Testing Scalar Multiplication on 3D Tensor:\n");
    
    Tensor scaled_3d = Scalar_Multiplication(tensor3d, test_3d_scalar);
    
    printf("Result (3D Tensor * 2.0):\n");
    
    Tensor_Printer(scaled_3d);

    printf("Testing 4D Tensor:\n");
    Tensor tensor4d = Tensor_Initializer(4, test_shape_4d, test_values_4d);
    
    printf("Original 4D Tensor (2x2x2x3):\n");
    
    Tensor_Printer(tensor4d);

    printf("Testing Batched Matrix Multiplication:\n");
    
    Tensor batch_a = Tensor_Initializer(3, test_batch_shape_a, test_batch_a_vals);
    
    Tensor batch_b = Tensor_Initializer(3, test_batch_shape_b, test_batch_b_vals);
    
    printf("Batch A (two 2x3):\n");
    Tensor_Printer(batch_a);

    printf("Batch B (2 3x2):\n");
    Tensor_Printer(batch_b);
    Tensor batch_result = Tensor_Multiplier(batch_a, batch_b);

    printf("Result of Batched Matrix Multiplication:\n");
    Tensor_Printer(batch_result);

    printf("Testing Tensor Addition with Same Shapes:\n");
    Tensor tensor3d_copy = Tensor_Initializer(3, test_shape_3d, test_values_3d);

    Tensor added_3d = Tensor_Adder(tensor3d, tensor3d_copy);
    printf("Result (3D Tensor + 3D Tensor, element-wise):\n");
    Tensor_Printer(added_3d);

    printf("Freeing memory :)\n");
    Tensor_Freer(matrix1);
    Tensor_Freer(matrix2);
    Tensor_Freer(matrix3);
    Tensor_Freer(matrix4);
    Tensor_Freer(matrix5);
    Tensor_Freer(vector1);
    Tensor_Freer(vector2);
    Tensor_Freer(tensor3d);
    Tensor_Freer(scaled_3d);
    Tensor_Freer(tensor4d);
    Tensor_Freer(batch_a);
    Tensor_Freer(batch_b);
    Tensor_Freer(batch_result);
    Tensor_Freer(tensor3d_copy);
    Tensor_Freer(added_3d);
    
    printf("Done\n");

    return 0;
}
