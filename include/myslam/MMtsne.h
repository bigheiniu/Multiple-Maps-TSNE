/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef MMTSNE_H
#define MMTSNE_H

#include <string>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class MMTSNE
{
public:
    void run(double *X, int N, int D, double* Y, double* weight, int no_dims,int no_maps, double perplexity, double theta, int rand_seed,
        bool skip_random_init, int max_iter=1000, int stop_lying_iter=250, int mom_switch_iter=250);
    bool load_data(double** data, int* n, int* d, int* no_dims,int* no_maps, double* theta, double* perplexity, int* rand_seed, int* max_iter);


private:
    void computedCdY(double *QQ, double *QZ,double * PQ, double *numerator, double *Y, int N, int D,int no_maps, double *dCdY);
    void computedCdW(double* PQ, double* QZ, double* proportions, double* numertor, double* dCdW, int N, int D, int no_maps);
    void computeNumerator(double* X, int N, int D, int no_maps,double* numerator);
    void computeQQ_QZ(double* numerator, double* proportions, double* QQ, double* QZ, int N, int D, int no_maps);
    void computePQ(double* P , double* Q, double* PQ, int N);
    void computeQ(double* Q, double* QZ, int N);
    void computeProportions(double* weight, double* proportions, int N, int no_maps);
    double evaluateNPR(double* P, double* Y, double* weight,int N, int D, int no_maps);
    void computeGradient(double *P, double *Y, double *weight, double* dCdW, double* dCdY, int N, int D, int no_maps);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
    double randn();
    void zeroMean(double* X, int N, int D);
    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
    void saveData(double* data, int row, int columns,std::string fileName);
    bool loadData(double* data, int row, int columns,std::string fileName);
};

#endif
