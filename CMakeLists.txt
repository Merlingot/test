cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()

# OPENCV 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# PYTHON
find_package(Python3 COMPONENTS Interpreter NumPy Development)
message("Python3_FOUND:${Python3_FOUND}")
message("Python3_NumPy_FOUND:${Python3_NumPy_FOUND}")
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries( ${PROJECT_NAME} ${Python3_LIBRARIES} Python3::NumPy)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

configure_file(pythonScript.py pythonScript.py COPYONLY)
