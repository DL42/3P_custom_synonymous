# Description of Mac/Linux/Unix Makefile for example_dadi. 
#
#############################
# build_path := Where to build program and put executable (note: folder must already exist)
# api_path_source := Location of API source folder
# api_path_include := Location of API include folder
# EXEC_FILE := Name of executable 
#
# NVCC := Compiler path, in this case nvcc is in $PATH
# CFLAGS := Compiler Flags: optimize most, fast math, add API include folder to include search path, equivalent to --relocatable-device-code=true --compile
# CODE := GPU types for which to build explicitly (I have a NVIDIA GTX 780M and 980) https://developer.nvidia.com/cuda-gpus, creates machine code for code=sm_30 (780) and code=sm_52 (980) and virtual architectures for all other generations which can be compiled JIT - code=compute_30 for generations between (3.0,5.0) and code=compute_50 for generations (5.0 and up) http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation
#
# object_names := Objects required for executable
# objects := Prepends build path to object names
#
# all := Command 'make' or 'make all' builds executable file $(EXEC_FILE) in location $(build_path)/
#
# ##### Object Dependencies Lists #####
# If one of the files on the right hand side of the : changes, the corresponding object must be recompiled.
# This is a developer's makefile - it assumes there will be changes to the GOFish API files, if not
# all the non-*.cu (non source file) dependencies are unnecessary.
# ##### End Object Dependencies Lists #####
#
# $(objects) := Make target all objects
# 	Compile source code into objects, $< := dependencies from Object Dependencies Lists, $@ := object in $(objects)
#
# $(build_path)/$(EXEC_FILE) := Make target executable EXEC_FILE which depends on all objects
# 	Link objects into executable EXEC_FILE, $@ := $(build_path)/$(EXEC_FILE)
#
# .PHONY := Defines 'all' and 'clean' as not true targets (i.e. don't remake executable if can't find files called 'all' or 'clean')
#
# clean := Action to perform for command 'make clean'
# 	Remove all objects and EXEC_FILE from build_path
#############################

build_path = ../bin
api_path_source = ../../3P/_internal
api_path_include = ../../3P
EXEC_FILE = GOFish

NVCC = nvcc -std=c++11
CFLAGS = -O3 --use_fast_math -I $(api_path_include)/ -dc
CODE = --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_30,code=compute_30 --generate-code arch=compute_52,code=compute_52

object_names = run.o spectrum.o
objects = $(addprefix $(build_path)/,$(object_names))

all:$(build_path)/$(EXEC_FILE)

##### OBJECT DEPENDENCIES #####
$(build_path)/run.o: run.cu $(api_path_source)/rng.cuh $(api_path_source)/cuda_helper_fun.cuh $(api_path_source)/go_fish_impl.cuh $(api_path_include)/go_fish_data_struct.h $(api_path_include)/go_fish.cuh $(api_path_include)/spectrum.h
$(build_path)/spectrum.o: $(api_path_source)/spectrum.cu $(api_path_include)/spectrum.h $(api_path_source)/cuda_helper_fun.cuh $(api_path_include)/go_fish_data_struct.h
##### END OBJECT DEPENDENCIES #####

$(objects):
	$(NVCC) $(CODE) $(CFLAGS) $< -o $@

$(build_path)/$(EXEC_FILE): $(objects)

	$(NVCC) $(CODE) $(objects) -o $@

.PHONY: all clean

clean:
	rm -f $(objects) $(build_path)/$(EXEC_FILE)