CC=gcc
LD=gcc
#CFLAGS=-ggdb -Wall -pedantic -std=gnu99 -O3
CFLAGS=-Wall -std=gnu99 -O3
LDFLAGS=

EX1_PROG=matrixMultiply

EX2_PROG=transpose

PROGS=$(EX2_PROG) $(EX1_PROG)

all: $(PROGS)

$(EX1_PROG):
	$(CC) -o $(EX1_PROG) $(CFLAGS) $(EX1_PROG).c
$(EX2_PROG):
	$(CC) -o $(EX2_PROG) $(CFLAGS) $(EX2_PROG).c

ex1:
	./$(EX1_PROG)

ex2:
	./$(EX2_PROG) 1000 10
	
clean:
	-rm -rf core *.o *~ "#"*"#" Makefile.bak $(PROGS) *.dSYM
