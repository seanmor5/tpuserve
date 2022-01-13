
all: tpuserve.cc libtpu.h
	g++ tpuserve tpuserve.cc -ldl