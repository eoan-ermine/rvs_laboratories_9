cmake_minimum_required(VERSION 3.27)
project(rvs_laboratories_8 LANGUAGES CXX CUDA)

add_executable(dataset_generator dataset_generator/dataset_generator.cpp)
add_executable(rvs_laboratories_8 rvs_laboratories_8/rvs_laboratories_8.cu)
