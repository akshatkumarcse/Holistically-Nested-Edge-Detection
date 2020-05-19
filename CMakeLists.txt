# Use this to while building: cmake -DOpenCV_DIR="C:\opencv\build\x64\vc15\lib"
# cmake --build . --config Release


cmake_minimum_required(VERSION 3.5)     # Min cmake version
project(HED)    # Name of the project i.e., the root folder name
find_package(OpenCV REQUIRED)   # Packages used in source code, in this case, OpenCV

set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(${OpenCV_INCLUDE_DIRS})     # What directories to include
add_executable(HED hed.cpp)     # HED is the name of Project and hed.cpp is the main file name
target_link_libraries(HED ${OpenCV_LIBS})

