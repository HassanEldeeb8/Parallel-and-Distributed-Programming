#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define N 2000000
#define K 5
#define MAX_ITER 20

using namespace std;

int main() {

    float* data = new float[N];
    float centroids[K];
    int* labels = new int[N];

    srand(time(NULL));

    for (int i = 0;i < N;i++)
        data[i] = rand() % 1000;

    for (int i = 0;i < K;i++)
        centroids[i] = rand() % 1000;

    clock_t start = clock();

    for (int iter = 0; iter < MAX_ITER; iter++) {

        for (int i = 0;i < N;i++) {

            float minDist = 1e20;
            int bestCluster = 0;

            for (int j = 0;j < K;j++) {

                float dist = fabs(data[i] - centroids[j]);

                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            labels[i] = bestCluster;
        }

        float sum[K] = { 0 };
        int count[K] = { 0 };

        for (int i = 0;i < N;i++) {
            sum[labels[i]] += data[i];
            count[labels[i]]++;
        }

        for (int j = 0;j < K;j++) {
            if (count[j] != 0)
                centroids[j] = sum[j] / count[j];
        }
    }

    clock_t end = clock();

    double time = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "CPU Execution Time: " << time << " seconds" << endl;

    delete[] data;
    delete[] labels;

    return 0;
}