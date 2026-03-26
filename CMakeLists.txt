cmake_minimum_required(VERSION 3.18)
project(cxx_ppo_agent CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Libtorch
set(Torch_DIR "C:/libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Source Files
Set(SOURCES
    src/main.cpp
    src/environments/CartPoleEnv.cpp
    src/environments/GymEnv.cpp
    src/networks/ActorCritic.cpp
    src/core/PPOAgent.cpp
)

# Executable
add_executable(cxx_ppo_agent ${SOURCES})

# Include Directories
target_include_directories(cxx_ppo_agent PRIVATE src)

# Link Libraries
target_link_libraries(cxx_ppo_agent
    ${TORCH_LIBRARIES}
    pybind11::embed
    Python3::Python
    nlohmann_json::nlohmann_json
)

# Copy LibTorch DLLs to output directory
add_custom_command(TARGET cxx_ppo_agent POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/config.json"
    "$<TARGET_FILE_DIR:cxx_ppo_agent>"
)
if (WIN32)
    file(GLOB TORCH_DLLS "${TORCH_DIR}/../../../lib/*.dll")
    foreach (DLL ${TORCH_DLLS})
        add_custom_command(TARGET cxx_ppo_agent POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLLS}
            $<TARGET_FILE_DIR:cxx_ppo_agent>
        )
    endforeach()
endif()