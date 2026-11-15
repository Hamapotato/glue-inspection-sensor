cmake_minimum_required(VERSION 3.24)
project(glue_train_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(train_cpp
    src/main.cpp
)
target_link_libraries(train_cpp PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(train_cpp PRIVATE /O2 /fp:fast)
