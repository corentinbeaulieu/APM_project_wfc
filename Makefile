CC := clang
NVCC := nvcc
IDIR := inc
SDIR := src
CFLAGS := -std=c17 -march=native -g -O3 -funroll-loops -fopenmp -I$(IDIR) -DWFC_CUDA
NVFLAGS := -g -O3 -I$(IDIR)
LDFLAGS := -lm -fopenmp -lcudart
TARGETS := wfc
OBJ := $(patsubst %.c,%.o,$(wildcard $(SDIR)/*.c $(SDIR)/solvers/*.c))
OBJ_CUDA := $(patsubst %.cu,%.cu.o,$(wildcard $(SDIR)/*.cu $(SDIR)/solvers/*.cu))

.SUFFIXES: .cu .cu.o .c .o

all: $(TARGETS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

%.cu.o: %.cu
	$(NVCC) $(NVFLAGS) -x cu -dc -o $@ $<

dlink.o: $(OBJ_CUDA)
	$(NVCC) $(NVFLAGS) -dlink -o $@ $^

wfc: $(OBJ) $(OBJ_CUDA) dlink.o
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJ) $(OBJ_CUDA) dlink.o

clear: clean
	rm -f $(TARGETS)
