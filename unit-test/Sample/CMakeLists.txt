cmake_minimum_required(VERSION 3.9)

project(SampleTest)

set(SAMPLE_INCS
${PROJECT_SOURCE_DIR}/include/SampleTest.h
)

set(SAMPLE_SRCS
${PROJECT_SOURCE_DIR}/src/SampleTest.cpp
${PROJECT_SOURCE_DIR}/src/main.cpp
)

add_executable(SampleTest ${SAMPLE_SRCS} ${SAMPLE_INCS})

target_include_directories(${PROJECT_NAME} PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(SampleTest hvr_sample gtest_main)

set_property(TARGET SampleTest PROPERTY FOLDER ${CMAKE_PROJECT_NAME}/unit-test)
