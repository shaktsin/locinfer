cmake_minimum_required(VERSION 3.20)

project(locinf LANGUAGES CXX)

add_executable(locinf 
    src/main.cpp
    src/model_loader.cpp
)

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(locinf PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(locinf PRIVATE cxx_std_20)
set_target_properties(locinf PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(locinf PRIVATE -Wall -Wextra -Wpedantic)
endif()