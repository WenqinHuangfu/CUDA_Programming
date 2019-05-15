#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

/* The naive transpose function as a reference. */
void transpose_naive(int n, int blocksize, int *dst, int *src) {
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            dst[y + x * n] = src[x + y * n];
        }
    }
}

/* Implement cache blocking below. You should NOT assume that n is a
 * multiple of the block size. */
void transpose_blocking(int n, int blocksize, int *dst, int *src) {
    // YOUR CODE HERE
    
    int regular_block = n / blocksize;
    int remain_block = n - regular_block * blocksize;
    
    // Transpose regular blocks
    // Block Row
    for (int i = 0; i < regular_block; i++)
    	// Block Column
    	for (int j = 0; j < regular_block; j++)
    		// Inner Row
    		for (int k = 0; k < blocksize; k++)
    			// Inner Column
    			for (int l = 0; l < blocksize; l++)
    				dst[(i * blocksize + k) * n + (j * blocksize + l)] = src[(j * blocksize + l) * n + (i * blocksize + k)];
    				
    // Transpose remain blocks
    // Two slices
    // Right slice
    for (int i = 0; i < regular_block; i++)
    	for (int j = 0; j < blocksize; j++)
	    	for (int k = 0; k < remain_block; k++)
	    		dst[(i * blocksize + j) * n + (blocksize * regular_block + k)] = src[(blocksize * regular_block + k) * n + (i * blocksize + j)];
	    		
	   // Down slice
    for (int i = 0; i < regular_block; i++)
    	for (int j = 0; j < remain_block; j++)
	    	for (int k = 0; k < blocksize; k++)
	    		dst[(blocksize * regular_block + j) * n + (blocksize * i + k)] = src[(blocksize * i + k) * n + (blocksize * regular_block + j)];
    
    // One corner
    for (int i = 0; i < remain_block; i++)
    	for (int j = 0 ; j < remain_block; j++)
    		dst[(blocksize * regular_block + i) * n + (blocksize * regular_block + j)] = src[(blocksize * regular_block + j) * n + (blocksize * regular_block + i)];
} 

void benchmark(int *A, int *B, int n, int blocksize, 
    void (*transpose)(int, int, int*, int*), char *description) {
 
    int i, j;
    printf("Testing %s: ", description);

    /* initialize A,B to random integers */
    srand48( time( NULL ) );
    for( i = 0; i < n*n; i++ ) A[i] = lrand48( );
    for( i = 0; i < n*n; i++ ) B[i] = lrand48( );

    /* measure performance */
    struct timeval start, end;

    gettimeofday( &start, NULL );
    transpose( n, blocksize, B, A );
    gettimeofday( &end, NULL );

    double seconds = (end.tv_sec - start.tv_sec) +
        1.0e-6 * (end.tv_usec - start.tv_usec);
    printf( "%g milliseconds\n", seconds*1e3 );


    /* check correctness */
    for( i = 0; i < n; i++ ) {
        for( j = 0; j < n; j++ ) {
            if( B[j+i*n] != A[i+j*n] ) {
                printf("Error!!!! Transpose does not result in correct answer!!\n");
                exit( -1 );
            }
        }
    }
}

int main( int argc, char **argv ) {
    if (argc != 3) {
        printf("Usage: transpose <n> <blocksize>\nExiting.\n");
        exit(1);
    }

    int n = atoi(argv[1]);
    int blocksize = atoi(argv[2]);

    /* allocate an n*n block of integers for the matrices */
    int *A = (int*)malloc( n*n*sizeof(int) );
    int *B = (int*)malloc( n*n*sizeof(int) );

    /* run tests */
    benchmark(A, B, n, blocksize, transpose_naive, "naive transpose");
    benchmark(A, B, n, blocksize, transpose_blocking, "transpose with blocking");

    /* release resources */
    free( A );
    free( B );
    return 0;
}

