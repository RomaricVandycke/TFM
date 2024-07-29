#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

// Define some constants
#define N 1000       // Number of samples (adjust as necessary)
#define D 42         // Number of features
#define EPOCHS 1000  // Number of epochs for training
#define LR 0.01      // Learning rate

// Function to calculate the sigmoid
__device__ float sigmoid(float z) {
    return 1.0 / (1.0 + expf(-z));
}

// Kernel function for computing predictions
__global__ void compute_predictions(float *X, float *weights, float *preds, int n, int d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float z = 0.0;
        for (int j = 0; j < d; ++j) {
            z += X[idx * d + j] * weights[j];
        }
        preds[idx] = sigmoid(z);
    }
}

// Kernel function for updating weights
__global__ void update_weights(float *X, float *y, float *weights, float *preds, int n, int d, float lr) {
    int j = threadIdx.x;
    if (j < d) {
        float gradient = 0.0;
        for (int i = 0; i < n; ++i) {
            gradient += (preds[i] - y[i]) * X[i * d + j];
        }
        weights[j] -= lr * gradient / n;
    }
}

// Function to read CSV file
void read_csv(const char *filename, float *X, float *y, int n, int d) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[2048];
    int row = 0, col = 0;
    while (fgets(line, sizeof(line), file) && row < n) {
        if (row > 0) {  // Skip the header row
            char *token = strtok(line, ",");
            col = 0;
            while (token) {
                if (col < d) {
                    X[(row - 1) * d + col] = atof(token);
                } else {
                    y[row - 1] = atof(token);
                }
                token = strtok(NULL, ",");
                col++;
            }
        }
        row++;
    }
    fclose(file);
}

int main() {
    float *h_X, *h_y, *h_weights, *h_preds;
    float *d_X, *d_y, *d_weights, *d_preds;

    // Allocate host memory
    h_X = (float*)malloc(N * D * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));
    h_weights = (float*)malloc(D * sizeof(float));
    h_preds = (float*)malloc(N * sizeof(float));

    // Initialize weights to small random values
    for (int i = 0; i < D; ++i) {
        h_weights[i] = (float)rand() / RAND_MAX - 0.5;
    }

    // Read data from CSV file
    read_csv("data.csv", h_X, h_y, N, D);

    // Allocate device memory
    cudaMalloc((void**)&d_X, N * D * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_weights, D * sizeof(float));
    cudaMalloc((void**)&d_preds, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_X, h_X, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, D * sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        compute_predictions<<<numBlocks, blockSize>>>(d_X, d_weights, d_preds, N, D);
        update_weights<<<1, D>>>(d_X, d_y, d_weights, d_preds, N, D, LR);
    }

    // Copy final weights back to host
    cudaMemcpy(h_weights, d_weights, D * sizeof(float), cudaMemcpyDeviceToHost);

    // Print final weights
    printf("Final weights:\n");
    for (int i = 0; i < D; ++i) {
        printf("%f ", h_weights[i]);
    }
    printf("\n");

    // Free memory
    free(h_X);
    free(h_y);
    free(h_weights);
    free(h_preds);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_weights);
    cudaFree(d_preds);

    return 0;
}
