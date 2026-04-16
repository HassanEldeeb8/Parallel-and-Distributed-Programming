#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 2000000
#define K 5
#define MAX_ITER 20

int main() {

    float *data = (float*)malloc(N * sizeof(float));
    float *centroids = (float*)malloc(K * sizeof(float));
    int *labels = (int*)malloc(N * sizeof(int));
    int *counts = (int*)malloc(K * sizeof(int));

    srand(time(NULL));

    // Initialize data
    for (int i = 0; i < N; i++)
        data[i] = rand() % 1000;

    for (int i = 0; i < K; i++)
        centroids[i] = rand() % 1000;

    clock_t start = clock();

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // Reset counts and centroid sums
        for (int i = 0; i < K; i++) {
            counts[i] = 0;
            centroids[i] = 0;
        }

        // Step 1: Assign clusters
        for (int i = 0; i < N; i++) {
            float minDist = 1e20;
            int bestCluster = 0;

            for (int j = 0; j < K; j++) {
                float dist = fabs(data[i] - centroids[j]);

                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            labels[i] = bestCluster;
        }

        // Step 2: Update centroids
        for (int i = 0; i < N; i++) {
            int label = labels[i];
            centroids[label] += data[i];
            counts[label]++;
        }

        // Step 3: Compute averages
        for (int i = 0; i < K; i++) {
            if (counts[i] > 0)
                centroids[i] /= counts[i];
        }
    }

    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    printf("CPU Execution Time: %f seconds\n", time);

    free(data);
    free(centroids);
    free(labels);
    free(counts);

    return 0;
}
