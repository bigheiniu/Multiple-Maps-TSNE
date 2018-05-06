//
// Created by 李一川 on 01/05/2018.
//

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <fstream>
#include "MMtsne.h"
#ifndef EPS
#define EPS 1e-8
#endif




using namespace std;



void MMTSNE::run(double *X, int N, int D, double* Y, double* weight, int no_dims,int no_maps, double perplexity, double theta, int rand_seed,
                 bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter) {

    // Set random seed
    if (skip_random_init != true) {
        if(rand_seed >= 0) {
            printf("Using random seed: %d\n", rand_seed);
            srand((unsigned int) rand_seed);
        } else {
            printf("Using current time as random seed...\n");
            srand(time(NULL));
        }
    }



    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }
    printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
    bool exact = (theta == .0) ? true : false;


    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
    double momentum = .5, final_momentum = .8;
    double eta = 200.0;
    double etw = 100.0;

    // Allocate some memory
    double* dCdY    = new double[N * no_maps * no_dims];
    double* uY    = new double[N * no_dims * no_maps ];
    double* gains = new double [N * no_dims * no_maps];
    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;
    double* dCdW = new double [N * no_maps];
    if(dCdY == NULL || dCdW == NULL || uY==NULL) { printf("Memory allocation failed!\n"); exit(1); }


    // Normalize input data (to prevent numerical problems)
    printf("Computing input similarities...\n");
    start = clock();
    zeroMean(X, N, D);
    double max_X = .0;
    for(int i = 0; i < N * D; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
    }
    for(int i = 0; i < N * D; i++) X[i] /= max_X;

    // Compute input similarities for exact t-SNE
    double* P;
    // Compute similarities
    P = new double[N * N];
    if(P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeGaussianPerplexity(X, N, D, P, perplexity);

    // Symmetrize input similarities
    printf("Symmetrizing...\n");
    int nN = 0;
    for(int n = 0; n < N; n++) {
            int mN = (n + 1) * N;
            for(int m = n + 1; m < N; m++) {
                P[nN + m] += P[mN + n];
                P[mN + n]  = P[nN + m];
                mN += N;
            }
            nN += N;
        }
    double sum_P = .0;
    for(int i = 0; i < N * N; i++) sum_P += P[i];
    for(int i = 0; i < N * N; i++) P[i] /= sum_P;

    // Initialize solution (randomly)
    //TODO: 为了测试程序
    for(int i = 0; i < N * no_maps * no_dims; ++i) Y[i] = randn() * .0001;
    for(int j = 0; j < N * no_maps; ++j) weight[j] = randn() * .0001;
//    loadData(Y,N*no_maps,no_dims,"./Y.dat");
    // Perform main training loop
    end = clock();
    printf("max_iter is %i",max_iter);
    printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    start = clock();
    for(int iter = 0; iter < max_iter; iter++) {

        // Compute (approximate) gradient
        computeGradient(P,Y,weight,dCdW,dCdY,N,no_dims,no_maps);

        // Update gains

//         Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims*no_maps; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dCdY[i];
        for(int i = 0; i < N * no_dims*no_maps; i++)  Y[i] = Y[i] + uY[i];



        for(int i = 0; i < N * no_dims * no_maps; i++) uY[i] = momentum * uY[i] - eta * dCdY[i];
        for(int i = 0; i < N * no_dims * no_maps; ++i) Y[i] = Y[i] + uY[i];
        for(int i = 0; i < N * no_maps; ++i) weight[i] = weight[i] - etw * dCdW[i];
        // Make solution zero-mean
        zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
            for(int i = 0; i < N * N; i++)        P[i] /= 12.0;
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if (iter > 0 && (iter % 10 == 0 || iter == max_iter - 1)) {
            end = clock();
            double NPR = .0;
            NPR = evaluateNPR(P, Y, weight,N,no_dims, no_maps);
            if(iter == 0)
                printf("Iteration %d: NPR is %f\n", iter + 1, NPR);
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: NPR is %f (100 iterations in %4.2f seconds)\n", iter, NPR, (float) (end - start) / CLOCKS_PER_SEC);
            }
            start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    delete dCdY;
    delete uY;
    delete dCdW;
    delete P;
    printf("Fitting performed in %4.2f seconds.\n", total_time);
}

// compute numertor: [1 + (|| y_i(m) - y_j(m) || ^ 2)] ^ -1
void MMTSNE::computeNumerator(double *X, int N, int D, int no_maps, double *numerator) {
    int N2 = N * N;
    const double* XnD = X;

    for (int n = 0; n < N; ++n, XnD += D) {
         for (int map = 0; map < no_maps; ++map) {
            const double *XmD = XnD + D;
            // numerator[N * N * no_maps]
            double *curr_elem = &numerator[n * N + map * N2 + n];
            *curr_elem = 0.0;
            double *curr_elem_sym = curr_elem + N;
            for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
                *(++curr_elem) = 0.0;
                for (int d = 0; d < D; ++d) {
                    *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
                }
                *curr_elem = 1.0 / (1.0  + *curr_elem);
                *curr_elem_sym = *curr_elem;
            }
        }
    }

//    saveData(numerator,N*no_maps,N,"./numertor.txt");
}


void MMTSNE::computeQQ_QZ(double *numerator, double *proportions, double *QQ, double *QZ, int N, int D,
                                          int no_maps) {
    //proportions[maps * N]
    //QQ[N, N, no_maps]
    //QZ[N,N] => 需要 calloc 初始为0
    //numerator => N * N * no_map
    const double* piNi = proportions;
    const double* numerNi = numerator;
    int N2 = N * N;
    for (int map = 0; map < no_maps; ++map) {
        for (int n = 0; n < N; ++n) {
            double *QQ_elem = &QQ[(N2 * map) + n * N + n];
            double *QQ_elem_sym= QQ_elem + N;
            double *QZ_elem = &QZ[n * N + n];
            double *QZ_elem_sym = QZ_elem + N;
            for (int m = n + 1; m < N; ++m, QQ_elem_sym += N, QZ_elem_sym+=N) {
                //保证对角线元素为0

                *(++QQ_elem) = (*(piNi + map * N + n)) * (*(piNi + map * N + m)) * (*(numerNi + N2 * map + N * n + m));
                *QQ_elem_sym = *QQ_elem;
                //m,n
                *(++QZ_elem) += *QQ_elem;
                *QZ_elem_sym = *QZ_elem;
            }
        }
    }
}

void MMTSNE::computePQ(double*P, double *Q, double *PQ, int N) {
    //P - Q
    for(int n = 0; n < N; ++n){
        //PQ 对角线直接为0
        double* PQ_elem = &PQ[n * N + n];
        double* PQ_elem_sym = PQ_elem + N;
        for(int m = n+1; m < N; ++m, PQ_elem_sym += N) {
            *(++PQ_elem)= (*(P + N * n + m)) - (*(Q +N * n + m));
            *PQ_elem_sym = *PQ_elem;
        }
    }
//    saveData(PQ,N,N,"PQ.txt");
}


void MMTSNE::computeQ(double *Q, double *QZ, int N) {
    double sum_q = .0;
    int N2 = N * N;
    for (int i = 0; i < N2; ++i) {
        sum_q += QZ[i];
    }

    for (int n = 0; n < N; ++n) {
        double* q_elem = &Q[n * N + n];
        *q_elem = 0.0;
        double* q_elem_sym = q_elem + N;
        const double* qz_elem = QZ + n * N + n;
        for (int m = n + 1; m < N; ++m, q_elem_sym += N) {
            *(++q_elem) = 0.0;
            ++qz_elem;
            *q_elem = *qz_elem / sum_q;
            *q_elem_sym = *q_elem;
        }
    }
}

void MMTSNE::computeProportions(double* weight, double* proportions, int N, int no_maps){
    // propotions[no_maps, N]
    //weight[no_maps, N]
    // calculate sum (exp(π_(i)_m))
    double* sum_pi = new double[ N ];
    const double* weight_const = weight;
    for (int map = 0; map < no_maps; ++map) {

        for (int i = 0; i < N; ++i) {

            *(proportions + map * N + i) = exp(-1 * (*(weight_const + N * map + i)));
            *(sum_pi + i) += *(proportions + map * N + i);
        }
    }

    for (int map = 0; map < no_maps; ++map) {
        for (int i = 0; i < N; ++i) {
            *(proportions + N * map + i) = (*(proportions + N * map + i)) / (*(sum_pi + i));
        }
    }

    delete sum_pi;
}


void MMTSNE::computedCdY(double *QQ, double *QZ, double*PQ, double *numerator, double *Y, int N, int D,int no_maps,
                                        double *dCdY) {
    int NxD = N * D;
    int N2 = N * N;
    for(int map = 0; map < no_maps; ++map) {
        for (int n =0; n < N; ++n) {
            for (int m = 0; m < N; ++m){
                if (m != n) {
                    double mult = 2.0 * QQ[N2*map+n*N+m] / QZ[n*N+m] * (PQ[n*N + m]) * numerator[N2 * map + n*N + m];
                    for (int d = 0; d < D; ++d) {
                        double Yn_m = Y[map*NxD + n*D + d] - Y[map * NxD + m*D + d];
                        dCdY[map*NxD + n*D +d ] += mult * Yn_m;
                    }
                }
            }
        }
    }
//    saveData(dCdY,N*no_maps,D,"dCdY.txt");
}



void MMTSNE::computedCdW(double *PQ, double *QZ, double *proportions, double *numertor, double *dCdW,
                                        int N, int D, int no_maps) {
    //TODO:先不考虑拉普拉斯正则以及 NAG 操作; => 比较通过跟 matlab 的程序进行比较, 使用 mnist 数据
    //dCdW:[no_maps * N]
    int N2 = N * N;
    for (int map = 0; map < no_maps; ++map) {
        for (int i =0; i < N; ++i) {
            double* dCdW_elem = &dCdW[map * N + i];
            *dCdW_elem = 0.0;
            for (int j = 0; j < N; ++j) {
                //不去计算对角线的值
                double midvalue = 0.0;
                if (i != j) {
                    for (int map_ex = 0; map_ex < no_maps; ++map_ex) {
                        double multi;
                        if (map_ex == map) {
                            multi = (proportions[N * map_ex + i] - 1.0);
                        } else {
                            multi = proportions[N * map_ex + i];
                        }
                        midvalue += multi * proportions[N * map_ex + j] * numertor[ N2 * map_ex + i * N + j];
                    }
                    double tmp = 2.0 /QZ[i * N + j] * (-1.0 *  PQ[i * N + j]);
                    *dCdW_elem += tmp * midvalue;
                }
            }
            *dCdW_elem = proportions[map * N + i] * (*dCdW_elem);
        }
    }
//    saveData(dCdW,no_maps,N,"dCdW.txt");
}

//目前仅仅比较一个最近邻

double MMTSNE::evaluateNPR(double* P, double* Y, double* weight,int N, int D, int no_maps) {
    int hit = 0;
    double* Q = new double[N * N];
    double* PQ = new double[N * N];
    double* QZ = new double[N * N];
    double* QQ = new double[N * N * no_maps];
    double* proportions = new double[no_maps * N];
    double* numerator = new double[N * N * no_maps];
    if(Q == NULL || PQ == NULL || QZ == NULL || QQ == NULL || proportions == NULL || numerator == NULL) {
        fprintf(stderr, "Memory allocation failed!\n"); exit(1);
    }
    //给 QZ 赋极小值
    for(int i = 0; i <N*N; ++i) QZ[i] = EPS;
    computeProportions(weight,proportions,N,no_maps);
    computeNumerator(Y,N,D,no_maps,numerator);
    computeQQ_QZ(numerator,proportions,QQ,QZ,N,D,no_maps);
    computeQ(Q,QZ,N);

    delete[] PQ;
    delete[] QZ;
    delete[] QQ;
    delete[] proportions;
    delete[] numerator;
    for(int i = 0; i < N; ++i) {
        double max_value_p = -1.0;
        double max_value_q = -1.0;
        int index_p = -1;
        int index_q = -1;
        for(int j = 0; j < N; ++j) {
            if (P[i * N + j] > max_value_p) {
                max_value_p = P[i * N + j];
                index_p = j;
            }
            if (Q[i * N + j] > max_value_q) {
                max_value_q = Q[i * N + j];
                index_q = j;
            }
        }
        if ((index_p == index_q) && (index_p != -1)) {
            hit +=1;
        }
    }
    delete[] Q;
    printf("his is %i \n",hit);
    return hit * 1.0 / N;
}








//计算 dCdW, dCdP 从而去修改 Y 和 weight 矩阵
void MMTSNE::computeGradient(double *P, double *Y, double *weight, double* dCdW, double* dCdY, int N, int D, int no_maps) {
    // Construct quadtree on current map
    //TODO: 中间变量太多, 占用内存可能会很大
    double* Q = new double[N * N];
    double* PQ = new double[N * N];
    double* QZ = new double[N * N];
    double* QQ = new double[N * N * no_maps];
    double* proportions = new double[no_maps * N];
    double* numerator = new double[N * N * no_maps];
    //TODO: 加上计算损失函数

    if (Q == NULL || PQ== NULL ||QZ== NULL ||QQ== NULL ||proportions== NULL  ||numerator==NULL ||PQ ==NULL) {
        fprintf(stderr, "Memory allocation failed!\n"); exit(1);
    }

    for(int i = 0; i < N * N; ++i) QZ[i] = EPS;
    computeProportions(weight,proportions,N,no_maps);
    computeNumerator(Y,N,D, no_maps, numerator);
    computeQQ_QZ(numerator,proportions,QQ,QZ,N,D,no_maps);
    computeQ(Q,QZ,N);
    computePQ(P,Q,PQ,N);
    computedCdY(QQ,QZ,PQ,numerator,Y,N,D,no_maps,dCdY);
    computedCdW(PQ,QZ,proportions,numerator,dCdW,N,D,no_maps);

    delete[] Q;
    delete[] PQ;
    delete[] QZ;
    delete[] QQ;
    delete[] proportions;
    delete[] numerator;

    //TODO: 返回cost
}

void MMTSNE::zeroMean(double* X, int N, int D) {

    // Compute data mean
    double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) {
        mean[d] /= (double) N;
    }

    // Subtract data mean
    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
        }
        nD += D;
    }
    free(mean); mean = NULL;
}

double MMTSNE::randn() {
    double x, y, radius;
    do {
        x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    y *= radius;
    return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool MMTSNE::load_data(double** data, int* n, int* d, int* no_dims,int* no_maps, double* theta, double* perplexity, int* rand_seed, int* max_iter) {
    // Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
    if((h = fopen("/Users/bigheiniu/course/graduate_pro/tsne/mmtsne/data.dat", "r+b")) == NULL) {
        printf("Error: could not open data file.\n");
        return false;
    }
    fread(n, sizeof(int), 1, h);											// number of datapoints
    fread(d, sizeof(int), 1, h);											// original dimensionality
    fread(theta, sizeof(double), 1, h);										// gradient accuracy
    fread(perplexity, sizeof(double), 1, h);								// perplexity
    fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
    fread(no_maps, sizeof(int),1,h);                                        // number of maps
    fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations
    *data = (double*) malloc(*d * *n * sizeof(double));
    if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data
    if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
    fclose(h);
    printf("Read the %i x %i data matrix successfully!\n", *n, *d);
    printf("no_dims is %i, no_maps is %i\n", *no_dims, *no_maps);
    printf("iterations is %i\n", *max_iter);
    printf("theta is %f\n",*theta);
    printf("perplexity is %f\n",*perplexity);
    return true;

}
void MMTSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(X, N, D, DD);

    // Compute the Gaussian kernel row by row
    int nN = 0;
    for(int n = 0; n < N; n++) {

        // Initialize some variables
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;
        double sum_P;

        // Iterate until we found a good perplexity
        int iter = 0;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < N; m++) P[n*N + m] = exp(-beta * DD[n*N + m]);
            P[n*N + n] = DBL_MIN;

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < N; m++) sum_P += P[n*N + m];
            double H = 0.0;
            for(int m = 0; m < N; m++) H += beta * (DD[n*N + m] * P[n*N + m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row normalize P
        for(int m = 0; m < N; m++) P[n*N + m] /= sum_P;
    }

    // Clean up memory
    free(DD);
}
void MMTSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;

    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

void MMTSNE::saveData(double *data, int row, int column, std::string fileName) {

    ofstream outFile(fileName);
    for(int i=0; i<row; i++){
        for (int j = 0; j <  column; ++j) {
            outFile<<data[i * column + j];
            outFile<<" ";
        }
        outFile<<"\n";
    }
}

bool MMTSNE::loadData(double* data, int row, int colum, std::string fileName) {
    FILE *h;
    if((h = fopen(fileName.c_str(), "r+b")) == NULL) {
        printf("Error: could not open data file.\n");
        return false;
    }
    fread(data, sizeof(double), row * colum, h);
    return true;
}