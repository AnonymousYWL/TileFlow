LIBXSMM_PREFIX = /home/nudt/ywl/MKL/MKL/2025.3#/home/nudt/intel/oneapi/2025.1
#LIBXSMM_PREFIX = /home/nudt/intel/oneapi/2025.1
LIBXSMM_INC    = $(LIBXSMM_PREFIX)/include
LIBXSMM_LIB    = $(LIBXSMM_PREFIX)/lib/intel64/libmkl_rt.so
LIBIOMP_LIB    = /home/nudt/ywl/MKL/MKL/2025.3/lib/libiomp5.so #/home/nudt/intel/oneapi/2025.1/lib/libiomp5.so
#LIBIOMP_LIB    = /home/nudt/intel/oneapi/2025.1/lib/libiomp5.so

OTHER_LIBS  = -lpthread -lm -ldl -lnuma

CC          = g++ -march=native
CFLAGS      = -O2 -I$(LIBXSMM_INC)
LINKER      = $(CC)

OBJS        = test.o

%.o: %.c
	 $(CC) $(CFLAGS) -fopenmp -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(CFLAGS) $(LIBIOMP_LIB) $(OTHER_LIBS) $(LIBXSMM_LIB) -o a.out

clean:
	rm -rf *.o a.out
