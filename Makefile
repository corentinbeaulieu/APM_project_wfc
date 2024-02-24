CC := clang
IDIR := inc
SDIR := src
CFLAGS := -std=c17 -march=native -g -O3 -funroll-loops -pthread -fopenmp -I$(IDIR)
# CFLAGS += -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60
LDFLAGS := -lm -fopenmp # -rt
TARGETS := wfc
OBJ := $(patsubst %.c,%.o,$(wildcard $(SDIR)/*.c $(SDIR)/solvers/*.c))

all: $(TARGETS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

wfc: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJ)

clear: clean
	rm -f $(TARGETS)
