cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(tpuserve)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS c_src/third_party/xla_data.proto c_src/third_party/hlo.proto)

file(GLOB TPUSERVE_SRCS "${C_SRC}/*.cc" "${C_SRC}/*.h" "${C_SRC}/third_party/*.h")

add_library(tpuserve SHARED ${TPUSERVE_SRCS} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(tpuserve ${Protobuf_LIBRARIES})

set_property(TARGET tpuserve PROPERTY CXX_STANDARD 17)
set_target_properties(tpuserve PROPERTIES PREFIX "lib")
set_target_properties(tpuserve PROPERTIES SUFFIX ".so")
set_target_properties(tpuserve PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(tpuserve PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -I${ERTS_INCLUDE_DIR} -O3 -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ldl")