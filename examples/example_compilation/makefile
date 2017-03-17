objects = main.o run.o spectrum.o shared.o go_fish_impl.o
build_path = ../../../build/example_compilation/
build_objects = $(addprefix $(build_path),$(objects))

all: $(objects)

	nvcc --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_32,code=sm_32 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50, --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_52,code=compute_52 $(build_objects) -o $(build_path)GOFish

%.o: %.cpp
	nvcc -O3 --use_fast_math -c $< -o $(build_path)$@

%.o: %.cu
	nvcc --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_32,code=sm_32 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_52,code=compute_52 -O3 --use_fast_math -I ../../include/ -dc $< -o $(build_path)$@

%.o: ../../source/%.cu
	nvcc --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_32,code=sm_32 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_52,code=compute_52 -O3 --use_fast_math -I ../../include/ -dc $< -o $(build_path)$@

clean:
	rm -f $(build_path)*.o $(build_path)GOFish