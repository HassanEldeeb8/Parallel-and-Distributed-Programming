#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

#define N 2000000
#define K 5
#define MAX_ITER 20

__global__ void assignClusters(float *data, float *centroids, int *labels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {

        float minDist = 1e20;
        int bestCluster = 0;

        for (int i = 0; i < K; i++) {

            float dist = fabs(data[idx] - centroids[i]);

            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }

        labels[idx] = bestCluster;
    }
}

__global__ void updateCentroids(float *data, int *labels, float *centroids, int *counts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {

        int label = labels[idx];

        atomicAdd(&centroids[label], data[idx]);
        atomicAdd(&counts[label], 1);
    }
}

int main() {

    float *data;
    float *centroids;
    int *labels;

    float *d_data;
    float *d_centroids;
    int *d_labels;
    int *d_counts;

    data = (float*)malloc(N * sizeof(float));
    centroids = (float*)malloc(K * sizeof(float));
    labels = (int*)malloc(N * sizeof(int));

    srand(time(NULL));

    for(int i = 0; i < N; i++)
        data[i] = rand() % 1000;

    for(int i = 0; i < K; i++)
        centroids[i] = rand() % 1000;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_centroids, K * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    clock_t start = clock();

    for(int iter = 0; iter < MAX_ITER; iter++) {

        cudaMemcpy(d_centroids, centroids, K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_counts, 0, K * sizeof(int));

        assignClusters<<<blocks, threads>>>(d_data, d_centroids, d_labels);

        cudaMemcpy(d_centroids, centroids, K * sizeof(float), cudaMemcpyHostToDevice);

        updateCentroids<<<blocks, threads>>>(d_data, d_labels, d_centroids, d_counts);

        cudaMemcpy(centroids, d_centroids, K * sizeof(float), cudaMemcpyDeviceToHost);

    }

    clock_t end = clock();

    float time = (float)(end - start) / CLOCKS_PER_SEC;

    printf("GPU Execution Time: %f seconds\n", time);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);

    free(data);
    free(centroids);
    free(labels);

    return 0;
}