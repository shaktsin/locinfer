cmake_minimum_required(VERSION 3.20)

project(locinf LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(locinf 
    src/main.cpp
    src/model_loader.cpp
    src/safetensors.cpp
    src/safetensor_checkpoint.cpp
)

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()
target_link_libraries(locinf PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(locinf PRIVATE cxx_std_20)
set_target_properties(locinf PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(locinf PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(checkpoint_test tests/checkpoint_test.cpp src/safetensor_checkpoint.cpp)
    target_include_directories(checkpoint_test PRIVATE src)
    target_link_libraries(checkpoint_test PRIVATE nlohmann_json::nlohmann_json)
    target_compile_features(checkpoint_test PRIVATE cxx_std_20)
    add_test(NAME checkpoint_load COMMAND checkpoint_test)
endif()
