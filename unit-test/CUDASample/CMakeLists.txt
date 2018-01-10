cmake_minimum_required(VERSION 3.9)

project(CUDASampleTest)

if(MSVC)
  set(DISABLE "__pragma(warning(push,0))")
  set(ENABLE "__pragma(warning(pop))")
endif()

set(CUDASAMPLE_INCS
${PROJECT_SOURCE_DIR}/include/CUDASampleTest.h
)

set(CUDASAMPLE_SRCS
${PROJECT_SOURCE_DIR}/src/CUDASampleTest.cpp
${PROJECT_SOURCE_DIR}/src/main.cpp
)

add_executable(CUDASampleTest ${CUDASAMPLE_SRCS} ${CUDASAMPLE_INCS})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(${PROJECT_NAME} hvr_cudasample gtest_main)

set_property(TARGET CUDASampleTest PROPERTY
            FOLDER ${CMAKE_PROJECT_NAME}/unit-test)
