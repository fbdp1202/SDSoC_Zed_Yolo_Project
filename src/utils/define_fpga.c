#include "define_fpga.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sds_lib.h>

#define GEMM_WIDTH 128
#define GEMM_HEIGHT 128
#define GEMM_SIZE GEMM_WIDTH * GEMM_HEIGHT

typedef struct layer_sds_buffer {
	float **A;
	float **B;
	float *C;
	int numWA, numHA;
	int numWB, numHB;
	int biasWA, biasHA;
	int biasWB, biasHB;
}slayer;

slayer *lBuffer;

void make_sds_buffer(int ns)
{
	lBuffer = (slayer *)malloc(sizeof(slayer)*ns);
	if ( lBuffer == NULL ) {
		printf("malloc error\n"); exit(-1);
	}
	int i;
	for ( i = 0; i < ns; i++ ) {
		lBuffer[i] = {0};
	}
}

void delete_sds_buffer()
{
	int i, j;
	for ( i = 0; i < sizeof(lBuffer) / sizeof(slayer); i++ ) {
		if ( lBuffer[i].A != NULL ) {
			for ( j = 0; j < lBuffer[i].numWA*lBuffer[i].numHA; j++ ) {
				free(lBuffer[i].A[j]);
			}
			free(lBuffer[i].A);
		}
		if ( lBuffer[i].B != NULL){
			for ( j = 0; j < lBuffer[i].numWB*lBuffer[i].numHB; j++ ) {
				free(lBuffer[i].B[j]);
			}
			free(lBuffer[i].B);
		}
		if ( lBuffer[i].C != NULL ) free(lBuffer[i].C);
	}
	free(lBuffer);
}

void set_sds_buffer(int index, float *weight, int m, int n, int k)
{
	int i, j, x, y;

	lBuffer[index].numWA = k/GEMM_WIDTH + 1;
	lBuffer[index].numHA = m/GEMM_HEIGHT + 1;
	lBuffer[index].numWB = n/GEMM_HEIGHT + 1;
	lBuffer[index].numHB = k/GEMM_WIDTH + 1;

	lBuffer[index].biasWA = k%GEMM_WIDTH;
	lBuffer[index].biasHA = m%GEMM_HEIGHT;
	lBuffer[index].biasWB = n%GEMM_HEIGHT;
	lBuffer[index].biasHB = k%GEMM_WIDTH;

	int lnumWA = lBuffer[index].numWA;
	int lnumHA = lBuffer[index].numHA;
	int lnumWB = lBuffer[index].numWB;
	int lnumHB = lBuffer[index].numHB;

	lBuffer[index].A = (float **)malloc(sizeof(float *)*lnumWA*lnumHA);
	for ( i = 0; i < lnumWA * lnumHA; i++ ) {
		lBuffer[index].A[i] = (float *)sds_alloc(GEMM_SIZE, sizeof(float));
	}

	for ( j = 0; j < numHA-1; j++ ) {
		for ( i = 0; i < numWA-1; i++ ) {
			for ( y = 0; y < GEMM_HEIGHT; y++) {
				for ( x = 0; x < GEMM_WIDTH; x++) {
					int ay = i*numHA + j;
					int ax = y*GEMM_WIDTH + x;
					int widx = j*k*GEMM_HEIGHT + i*GEMM_WIDTH + y*k + x;
					lBuffer[index].A[ay][ax] = weight[widx];
				}
			}
		}
	}

	lBuffer[index].B = (float **)malloc(sizeof(float *)*lnumWB*lnumHB);
	for ( i = 0; i < lnumWB*lnumHB; i++ ) {
		lBuffer[index].B[i] = (float *)sds_alloc(sizeof(float)*GEMM_SIZE);
	}

	lBuffer[index].C = (float *)sds_alloc(sizeof(float)*GEMM_SIZE);
}
