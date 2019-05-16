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

const int TILE_WIDTH_GEMM = 16;
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

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

/* Question 1 */
// GPU based GEMM with SM-specific shared memory
__global__ void MatrixMultiplyKernel(const float* devM, const float* devN,float* devP, const int width){
	__shared__ float sM[TILE_WIDTH_GEMM][TILE_WIDTH_GEMM];
	__shared__ float sN[TILE_WIDTH_GEMM][TILE_WIDTH_GEMM];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = bx * TILE_WIDTH_GEMM + bx;
	int row = by * TILE_WIDTH_GEMM + ty;
	
	// Initialize accumulator to 0. Then multiply/add
	float pValue = 0;
	
	for (int m = 0; m < width / TILE_WIDTH_GEMM; m++) {
		sM[ty][tx] = devM[row *width+(m*TILE_WIDTH_GEMM + tx)];
		sN[ty][tx] = devN[col+(m *TILE_WIDTH_GEMM+ty)*width];
		__syncthreads();
		
		for (int k = 0; k < TILE_WIDTH_GEMM; ++k)
			pValue += sM[ty][k] * sN[k][tx];
		__syncthreads();
	}
	
	devP[row * width + col] = pValue;
}

/* Question 2 */
// Simple matrix copying
__global__ void copy(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

// Matrix copy with shared memory
__global__ void copySharedMem(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
  
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}

// Native transpose
__global__ void transposeNaive(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

// Coalesced transpose with block shared memory
__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Coalesced transpose with shared memory and matrix padding
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
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
    
    // HW2: Question 1
    int m_size1 = 1600, n_size1 = 1600;
    int width1 = 1600;
    int iterations1 = 100;
    float GFLOPs = 0;
    
    float *A_h1 = (float *)malloc( m_size1*n_size1*sizeof(float));
    float *B_h1 = (float *)malloc( m_size1*n_size1*sizeof(float));
    float *C_h1 = (float *)malloc( m_size1*n_size1*sizeof(float));

    float *A_d1, *B_d1, *C_d1;
    cudaMalloc((void**)&A_d1, m_size1*n_size1*sizeof(float));
    cudaMalloc((void**)&B_d1, m_size1*n_size1*sizeof(float));
    cudaMalloc((void**)&C_d1, m_size1*n_size1*sizeof(float));
    
    dim3 dimGrid1(1, 1, 1);
    dim3 dimBlock1(1, 1, 1);
    
    cudaMemcpy(A_d1, A_h1, m_size1*n_size1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d1, B_h1, m_size1*n_size1*sizeof(float), cudaMemcpyHostToDevice);
		
    for (int i = 0; i < iterations1; i++) {
        gettimeofday( &start, NULL );
        MatrixMultiplyKernel<<<dimGrid1, dimBlock1>>>(A_d1, B_d1, C_d1, width1);
        gettimeofday( &end, NULL );
			
        double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
        GFLOPs += 2e-9*width1*width1*width1/seconds;
    }
		
    cudaMemcpy(C_h1, C_d1, m_size1*n_size1*sizeof(float), cudaMemcpyDeviceToHost);

    GFLOPs /= iterations1;
		
    printf( "%.3f GFLOPs/s\n", GFLOPs );

    cudaFree( A_d1 );
    cudaFree( B_d1 );
    cudaFree( C_d1 );
    free( A_h1 );
    free( B_h1 );
    free( C_h1 );

    printf("\n\n");
	
    // HW2: Question 2
    int m_size2 = 1024, n_size2 = 1024;
    int width2 = 1024;
    int iterations2 = 100;
    float Mem_Acc_Rate[5] = {0};

    float *A_h2 = (float *)malloc( m_size2*n_size2*sizeof(float));
    float *B_h2 = (float *)malloc( m_size2*n_size2*sizeof(float));

    float *A_d2, *B_d2;
    cudaMalloc((void**)&A_d2, m_size2*n_size2*sizeof(float));
    cudaMalloc((void**)&B_d2, m_size2*n_size2*sizeof(float));
    
    dim3 dimGrid2(1, 1, 1);
    dim3 dimBlock2(1, 1, 1);
    
    // Simple matrix copying
    for (int i = 0; i < iterations2; i++) {
        gettimeofday( &start, NULL );
        copy<<<dimGrid2, dimBlock2>>>(A_d2, B_d2);
        gettimeofday( &end, NULL );
			
        double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
        Mem_Acc_Rate[0] += 2e-9*width1*width1*width1/seconds;
    }

    return 0;
}
