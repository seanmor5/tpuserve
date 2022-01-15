
all: tpuserve.cc libtpu.h logging.h model.cc model.h
	g++ -o tpuserve model.cc model.h tpuserve.cc -ldl