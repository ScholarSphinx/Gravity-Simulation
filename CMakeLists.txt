cmake_minimum_required(VERSION 3.20)
project(GravitySimulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

add_executable(GravitySimulation
    src/main.cpp
)

target_include_directories(GravitySimulation PRIVATE src)

target_link_libraries(GravitySimulation
    PRIVATE
        glfw
        OpenGL::GL
        GLEW::GLEW
        glm::glm
)
