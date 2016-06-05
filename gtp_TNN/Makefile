GENCODE=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_20,code=sm_20
OPT=-O3 -DCHECK_CORRECTNESS

NVCC_FLAGS   = $(GENCODE) $(OPT)
NVCC_LDFLAGS = $(NVCC_FLAGS) -lcublas -ltbb

MAIN_SRC = test_TNN_main.cu
DGEMM_SRC = $(wildcard test_dTNN*.cu)
SGEMM_SRC = $(wildcard test_sTNN*.cu)
DGEMM_EXECUTABLE = $(subst .cu,,$(DGEMM_SRC))
SGEMM_EXECUTABLE = $(subst .cu,,$(SGEMM_SRC))

EXECUTABLE = $(DGEMM_EXECUTABLE) $(SGEMM_EXECUTABLE)

all: $(EXECUTABLE)

SHELL = /bin/bash

check: clean
	make OPT='$(OPT) -DCHECK_CORRECTNESS' all

test_dTNN_main.o: $(MAIN_SRC)
	nvcc $(NVCC_FLAGS) -c -DUSING_DOUBLE -o $@ $<

test_sTNN_main.o: $(MAIN_SRC)
	nvcc $(NVCC_FLAGS) -c -DUSING_FLOAT  -o $@ $<

%.o: %.cu
	. init_tbb.bash && nvcc $(NVCC_FLAGS) -c -o $@ $<

test_dTNN_%: test_dTNN_%.o test_dTNN_main.o
	. init_tbb.bash && nvcc $(NVCC_LDFLAGS) $^ -o $@

test_sTNN_%: test_sTNN_%.o test_sTNN_main.o
	. init_tbb.bash && nvcc $(NVCC_LDFLAGS) $^ -o $@

clean:
	$(RM) *.o $(EXECUTABLE) *.cpp.ii

.PHONY:
	all clean check
