cmake_minimum_required (VERSION 2.6)

project(digits)

add_executable(digits digits_main.cpp)

target_compile_definitions(digits PRIVATE ARMA_DONT_USE_WRAPPER)

target_compile_options(digits PRIVATE -O3 -march=native -fopenmp)

target_link_libraries(digits blas lapack omp)

install(TARGETS digits DESTINATION bin)

