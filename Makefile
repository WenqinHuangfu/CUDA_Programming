CC=nvcc
CFLAGS=
LDFLAGS=

EXEC = CUDA_Programming.x

all: $(EXEC)

#load cuda in the shell prompt
#       module load cuda

CUDA_Programming.x: CUDA_Programming.cu
	$(CC) -o $@ $^ $(LDFLAGS)

submitCUDA:
	sbatch -v CUDA_Programming.job

#.c.o:
#       $(CC)  $(CFLAGS) -c $<

clean:
	rm  $(EXEC)
