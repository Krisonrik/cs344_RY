############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cs344 LANGUAGES CXX CUDA)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(MSVC)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP /WX /sdl")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4127") # condition is constant
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44062")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44191")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44242")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44254")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44255")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44263")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44264")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44265")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44266")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44287")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44289")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44296")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44302")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44311")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44339")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44342")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44350")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44355")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44370")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44371")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44388")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44412")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44431")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44435")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44437")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44444")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44471")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44472")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44536")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44545")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44546")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44547")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44548")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44555")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44557")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44574")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44628")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44640")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44668")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44682")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44686")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44692")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44738")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44786")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44826")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44837")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44905")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44906")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44917")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44928")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44931")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44946")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44962")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44986")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44987")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44988")
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Werror -fsigned-char")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -freg-struct-return -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith -Wcast-qual -Wshadow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Werror -fsigned-char")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wpedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith -Wcast-qual -Wshadow")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_LIBRARIES})
link_libraries(${OpenCV_LIBS} ${CUDA_LIBRARIES})

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_WARNING "cross-execution-space-call")
  set(CUDA_NVCC_FLAGS "-ccbin /usr/bin/clang;") 
  # -gencode;arch=compute_30,code=sm_30;  
  # -gencode;arch=compute_35,code=sm_35;
  # -gencode;arch=compute_35,code=compute_35;
  # -gencode;arch=compute_20,code=sm_20; 
  # -gencode;arch=compute_11,code=sm_11; 
  # -gencode;arch=compute_12,code=sm_12;
  # -gencode;arch=compute_13,code=sm_13;

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_32")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_20,code=sm_20")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_12,code=sm_13")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_12,code=sm_12")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_11,code=sm_11")

  # new flags introduced in CUDA 9
  # set(CUDA_WARNING "reorder,cross-execution-space-call,deprecated-declarations")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror ${CUDA_WARNING} -restrict")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")


  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  # set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory (HW1)
  add_subdirectory (HW2)
  add_subdirectory (HW3)
  add_subdirectory (HW4)
  add_subdirectory (HW5)
  add_subdirectory (HW6)
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
