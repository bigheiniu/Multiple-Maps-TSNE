#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "MMtsne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
	int origN, N, D, no_dims, no_maps,max_iter, *landmarks;
	double perc_landmarks;
	double perplexity, theta, *data;
    int rand_seed = -1;
    MMTSNE* MMtsne = new MMTSNE();

    // Read the parameters and the dataset
	if(MMtsne->load_data(&data, &origN, &D, &no_dims, &no_maps, &theta, &perplexity, &rand_seed, &max_iter)) {

		// Make dummy landmarks
        N = origN;
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		double* Y = new double[N* no_maps * no_dims];
		double* weight = new double[N * no_maps];
        for (int i = 0; i < no_maps; ++i){
            for (int j = 0; j < N; j ++) {
                weight[i * N + j] = 1.0 / (double)no_maps;
            }
        }
//		double* costs = (double*) calloc(N, sizeof(double));
        if(Y == NULL ) { printf("Memory allocation failed!\n"); exit(1); }
		MMtsne->run(data, N, D, Y, weight, no_dims,no_maps,perplexity, theta, rand_seed, false, max_iter);

		// Save the results
//		tsne->save_data(Y, landmarks, costs, N, no_dims);

        // Clean up the memory
		free(data); data = NULL;
		delete Y;
		free(landmarks); landmarks = NULL;
		delete weight;
    }
    delete(MMtsne);
}
