PROJECT(opencv.torch)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

SET(BUILD_TESTS ON)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${OpenCV_INCLUDE_DIRS})

SET(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include")

ADD_LIBRARY(Common SHARED "${SOURCE_DIR}/Common.cpp")
TARGET_LINK_LIBRARIES(Common TH ${OpenCV_LIBS})

SET(SHARED_LIBS imgcodecs highgui features2d flann imgproc ml optflow photo superres video videoio)

FOREACH(LIB ${SHARED_LIBS})
    ADD_LIBRARY(${LIB} SHARED "${SOURCE_DIR}/${LIB}.cpp")
    TARGET_LINK_LIBRARIES(${LIB} Common)
ENDFOREACH()

if (BUILD_TESTS)
    file(GLOB Tests_SRC "${SOURCE_DIR}/tests/*.cpp")
    ADD_LIBRARY(Tests SHARED ${Tests_SRC})
    TARGET_LINK_LIBRARIES(Tests Common)
endif()