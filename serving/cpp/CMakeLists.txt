# serving/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(kvsched CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clang-tidy / editor tooling
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

option(KVSCHED_SANITIZE "Build with ASan + UBSan" OFF)

option(KVSCHED_PYTHON   "Build the pybind11 module"  ON)

if(KVSCHED_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(kvsched_core
  src/block.cpp
  src/scheduler.cpp
)
target_include_directories(kvsched_core PUBLIC include)
target_compile_options(kvsched_core PRIVATE -Wall -Wextra -Wpedantic)

# ---- tests ----
include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(
  test_kvsched
  tests/test_block.cpp
  tests/test_hash.cpp
)
target_link_libraries(test_kvsched PRIVATE kvsched_core GTest::gtest_main)
add_test(NAME kvsched COMMAND test_kvsched)

if(KVSCHED_PYTHON AND NOT KVSCHED_SANITIZE)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
 
  pybind11_add_module(_kvsched src/bindings.cpp)
  target_link_libraries(_kvsched PRIVATE kvsched_core)
endif()
