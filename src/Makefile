
EXECUTABLE := sudoku

CU_FILES   := cudoku.cu

CU_DEPS    := 

CC_FILES   := util.cpp sudoku.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudadevrt -lcudart 
NVCC=nvcc
NVCCFLAGS=-rdc=true -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

OBJS= $(OBJDIR)/util.o $(OBJDIR)/cudoku.o $(OBJDIR)/sudoku.o


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
