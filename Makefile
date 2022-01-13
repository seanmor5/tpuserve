
all: tpuserve.cc libtpu.h logging.h
	g++ -o tpuserve logging.h tpuserve.cc -ldl
