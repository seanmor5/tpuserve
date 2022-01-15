
all: tpuserve.cc libtpu.h logging.h model.cc model.h
	g++ -std=c++17 -o tpuserve model.cc model.h tpuserve.cc -ldl
