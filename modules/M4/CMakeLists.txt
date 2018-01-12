cmake_minimum_required(VERSION 3.9)

project(M4)

if(MSVC)
  set(DLLIMPORT "__declspec(dllimport)")
  set(DLLEXPORT "__declspec(dllexport)")
  set(DISABLE "__pragma(warning(push,0))")
  set(ENABLE "__pragma(warning(pop))")
endif()

set(PROJECT_SRCS
${PROJECT_SOURCE_DIR}/src/compare.cpp
${PROJECT_SOURCE_DIR}/src/HW4.cu
${PROJECT_SOURCE_DIR}/src/loadSaveImage.cpp
${PROJECT_SOURCE_DIR}/src/reference_calc.cpp
${PROJECT_SOURCE_DIR}/src/student_func.cu
)

set(PROJECT_INCS
${PROJECT_SOURCE_DIR}/include/Hvr/HW4/compare.h
${PROJECT_SOURCE_DIR}/include/Hvr/HW4/loadSaveImage.h
${PROJECT_SOURCE_DIR}/include/Hvr/HW4/reference_calc.h
${PROJECT_SOURCE_DIR}/include/Hvr/HW4/timer.h
${PROJECT_SOURCE_DIR}/include/Hvr/HW4/utils.h
)

source_group("include" FILES ${PROJECT_INCS})
source_group("src"     FILES ${PROJECT_SRCS})

find_package( OpenCV REQUIRED )

add_library(${PROJECT_NAME} SHARED ${PROJECT_SRCS} ${PROJECT_INCS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

target_compile_definitions(${PROJECT_NAME} PUBLIC
                          HVR_WINDOWS_DISABLE_ALL_WARNING=${DISABLE}
                          HVR_WINDOWS_ENABLE_ALL_WARNING=${ENABLE})

target_compile_definitions(${PROJECT_NAME} INTERFACE
                          M3_DLL=${DLLIMPORT})

target_compile_definitions(${PROJECT_NAME} PRIVATE
                          M3_DLL=${DLLEXPORT})

target_include_directories(${PROJECT_NAME} PUBLIC
                          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set_property(TARGET ${PROJECT_NAME} PROPERTY
            FOLDER ${CMAKE_PROJECT_NAME}/modules)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${CMAKE_PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib COMPONENT install-app
        LIBRARY DESTINATION lib COMPONENT install-app
        RUNTIME DESTINATION bin COMPONENT install-app)

install(FILES ${PROJECT_SOURCE_DIR}/include/Hvr/CUDASample/CUDASample.cuh
              ${PROJECT_SOURCE_DIR}/include/Hvr/CUDASample/CUDAConfig.h
        DESTINATION include/Hvr/CUDASample)

export(TARGETS ${PROJECT_NAME}
      APPEND FILE ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}Targets.cmake)

file(APPEND ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake
    "find_dependency(OpenCV REQUIRED)\n")
