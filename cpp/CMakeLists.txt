cmake_minimum_required(VERSION 3.18)
project(subject_matting VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime: either an official release directory (ONNXRUNTIME_DIR with include/ and lib/), or explicit
# ONNXRUNTIME_INCLUDE_DIR + ONNXRUNTIME_LIBRARY (e.g. headers + the dylib shipped in the onnxruntime pip wheel).
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime release directory")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_DIR}/include" "${ONNXRUNTIME_DIR}/include/onnxruntime" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort-headers")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime onnxruntime.1.31.0 HINTS "${ONNXRUNTIME_DIR}/lib")
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found: set ONNXRUNTIME_DIR, or ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY")
endif()

# The library: C API in include/subject_matting.h, everything else hidden.
add_library(subject_matting SHARED src/c_api.cpp src/pipeline.cpp src/image_ops.cpp)
target_include_directories(subject_matting PUBLIC include PRIVATE src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(subject_matting PRIVATE "${ONNXRUNTIME_LIBRARY}")
target_compile_definitions(subject_matting PRIVATE SM_VERSION="${PROJECT_VERSION}" SM_BUILDING_DLL)
set_target_properties(subject_matting PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  # Ship libonnxruntime next to libsubject_matting.
  BUILD_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
  INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>")

# Internal static library for the unit tests (they call C++ functions the shared library hides).
add_library(sm_core STATIC src/pipeline.cpp src/image_ops.cpp)
target_include_directories(sm_core PUBLIC src "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(sm_core PUBLIC "${ONNXRUNTIME_LIBRARY}")

add_executable(sm_cli tools/sm_cli.cpp)
target_include_directories(sm_cli PRIVATE third_party)
target_link_libraries(sm_cli PRIVATE subject_matting)
set_target_properties(sm_cli PROPERTIES INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@executable_path/../lib,$ORIGIN/../lib>")

add_executable(sm_ops_test tests/ops_test.cpp)
target_link_libraries(sm_ops_test PRIVATE sm_core)

install(TARGETS subject_matting sm_cli LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/subject_matting.h DESTINATION include)
# The ONNX Runtime library ships next to ours (found through @loader_path / $ORIGIN).
file(GLOB ONNXRUNTIME_SHARED_FILES "${ONNXRUNTIME_DIR}/lib/libonnxruntime*.dylib" "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so*")
install(FILES ${ONNXRUNTIME_SHARED_FILES} DESTINATION lib)
