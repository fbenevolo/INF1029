all:
	clear
	nvcc -o matrix_lib_test matrix_lib_test.cu matrix_lib.cu timer.c
	./matrix_lib_test 5 16 16 16 16 1024 floats_1024_2.0f.dat floats_1024_5.0f.dat result1.dat result2.dat