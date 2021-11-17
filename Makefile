COMPILER_FLAGS = -O3 -fopenmp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -std=c99 -march=native

winograd43: driver_43.c winograd_43.c
	g++ -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) driver_43.c winograd_43.c -o winograd43 $(COMPILER_FLAGS) -DMKL_DIRECT_CALL_JIT

winograd23: driver.c winograd.c
	g++ -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) driver.c winograd.c -o winograd23 $(COMPILER_FLAGS)


.PHONY : clean

clean:
	rm winograd43 winograd23
