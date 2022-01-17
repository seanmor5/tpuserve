PRIV_DIR = priv

CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -shared -std=c++11

LDFLAGS = -ldl

ARTIFACT = $(PRIV_DIR)/libtpuserve.so

$(ARTIFACT): c_src/tpuserve.cc c_src/tpuserve_driver.cc c_src/tpuserve_driver.h c_src/libtpu.h
	mkdir -p $(PRIV_DIR)
	$(CXX) $(CFLAGS) c_src/libtpu.h c_src/tpuserve_nif_util.h c_src/tpuserve_nif_util.cc c_src/tpuserve_driver.h c_src/tpuserve_driver.cc c_src/tpuserve.cc -o $(ARTIFACT) $(LDFLAGS)
