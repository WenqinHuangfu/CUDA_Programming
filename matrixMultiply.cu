#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>

/* To save you time, we are including all 6 variants of the loop ordering
   as separate functions and then calling them using function pointers.
   The reason for having separate functions that are nearly identical is
   to avoid counting any extraneous processing towards the computation
   time.  This includes I/O accesses (printf) and conditionals (if/switch).
   I/O accesses are slow and conditional/branching statements could
   unfairly bias results (lower cases in switches must run through more
   case statements on each iteration).
*/
void multMat1( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is ijk loop order. */
    for( i = 0; i < n; i++ )
        for( j = 0; j < n; j++ )
            for( k = 0; k < n; k++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat2( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is ikj loop order. */
    for( i = 0; i < n; i++ )
        for( k = 0; k < n; k++ )
            for( j = 0; j < n; j++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat3( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is jik loop order. */
    for( j = 0; j < n; j++ )
        for( i = 0; i < n; i++ )
            for( k = 0; k < n; k++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat4( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is jki loop order. */
    for( j = 0; j < n; j++ )
        for( k = 0; k < n; k++ )
            for( i = 0; i < n; i++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat5( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is kij loop order. */
    for( k = 0; k < n; k++ )
        for( i = 0; i < n; i++ )
            for( j = 0; j < n; j++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

void multMat6( int n, float *A, float *B, float *C ) {
    int i,j,k;
    /* This is kji loop order. */
    for( k = 0; k < n; k++ )
        for( j = 0; j < n; j++ )
            for( i = 0; i < n; i++ )
                C[i+j*n] += A[i+k*n]*B[k+j*n];
}

__global__ void MatrixMultiplyKernel(const float* devM, const float* devN,float* devP, const int width){
	int TILE_WIDTH = 16;
	
	__shared__ float sM[16][16];
	__shared__ float sN[16][16];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = bx * TILE_WIDTH + bx;
	int row = by * TILE_WIDTH + ty;
	
	// Initialize accumulator to 0. Then multiply/add
	float pValue = 0;
	
	for (int m = 0; m < width / TILE_WIDTH; m++) {
		sM[ty][tx] = devM[row *width+(m*TILE_WIDTH + tx)];
		sN[ty][tx] = devN[col+(m *TILE_WIDTH+ty)*width];
		__syncthreads();
		
		for (int k = 0; k < TILE_WIDTH; ++k)
			pValue += sM[ty][k] * sN[k][tx];
		__syncthreads();
	}
	
	devP[row * width + col] = pValue;
}

/* uses timing features from sys/time.h that you haven't seen before */
int main( int argc, char **argv ) {
    // CPU implementation
    int nmax = 1000,i;

    void (*orderings[])(int,float *,float *,float *) =
        {&multMat1,&multMat2,&multMat3,&multMat4,&multMat5,&multMat6};
    char *names[] = {"ijk","ikj","jik","jki","kij","kji"};

    float *A = (float *)malloc( nmax*nmax * sizeof(float));
    float *B = (float *)malloc( nmax*nmax * sizeof(float));
    float *C = (float *)malloc( nmax*nmax * sizeof(float));

    struct timeval start, end;

    /* fill matrices with random numbers */
    for( i = 0; i < nmax*nmax; i++ ) A[i] = drand48()*2-1;
    for( i = 0; i < nmax*nmax; i++ ) B[i] = drand48()*2-1;
    for( i = 0; i < nmax*nmax; i++ ) C[i] = drand48()*2-1;

    for( i = 0; i < 6; i++) {
        /* multiply matrices and measure the time */
        gettimeofday( &start, NULL );
        (*orderings[i])( nmax, A, B, C );
        gettimeofday( &end, NULL );

        /* convert time to Gflop/s */
        double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
        double Gflops = 2e-9*nmax*nmax*nmax/seconds;
        printf( "%s:\tn = %d, %.3f Gflop/s\n", names[i], nmax, Gflops );
    }
    
    free( A );
    free( B );
    free( C );
    
    printf("\n\n");
    
    // GPU implementation
    int m_size = 1600, n_size = 1600;
    int width = 1600;
    int iterations = 100;
    float Gflops = 0;
    
    float *A_h = (float *)malloc( m_size*n_size * sizeof(float));
    float *B_h = (float *)malloc( m_size*n_size * sizeof(float));
    float *C_h = (float *)malloc( m_size*n_size * sizeof(float));

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, m_size*n_size*sizeof(float));
    cudaMalloc((void**)&B_d, m_size*n_size*sizeof(float));
    cudaMalloc((void**)&C_d, m_size*n_size*sizeof(float));
    
    dim3 dimGrid(100, 100, 1);
    dim3 dimBlock(16, 16, 1);
    
    cudaMemcpy(A_d, A_h, m_size*n_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, m_size*n_size*sizeof(float), cudaMemcpyHostToDevice);
		
    for (int i = 0; i < iterations; i++) {
        gettimeofday( &start, NULL );
        MatrixMultiplyKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
        gettimeofday( &end, NULL );
			
        double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
        Gflops += 2e-9*nmax*nmax*nmax/seconds;
    }
		
    cudaMemcpy(C_h, C_d, m_size*n_size*sizeof(float), cudaMemcpyDeviceToHost);

    Gflops /= iterations;
		
    printf( "%s:\tn = %d, %.3f Gflop/s\n", names[i], nmax, Gflops );

    free( A_d );
    free( B_d );
    free( C_d );
    free( A_h );
    free( B_h );
    free( C_h );

    printf("\n\n");

    return 0;
}
