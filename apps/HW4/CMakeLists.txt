cmake_minimum_required(VERSION 3.9)

project(HW4)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(PROJECT_SRCS
${PROJECT_SOURCE_DIR}/src/main.cpp
)

source_group("src" FILES ${PROJECT_SRCS})

find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SRCS})

target_link_libraries(${PROJECT_NAME} M4 ${CUDA_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER ${CMAKE_PROJECT_NAME}/apps)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${CMAKE_PROJECT_NAME}Targets
        DESTINATION bin COMPONENT install-app)
