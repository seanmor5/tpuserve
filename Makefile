PRIV_DIR = priv

CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -shared

LDFLAGS = -ldl

ARTIFACT = $(PRIV_DIR)/libtpuserve.so

SRCS = c_src/libtpu.h c_src/logging.h c_src/tpuserve_nif_util.h c_src/tpuserve_nif_util.cc \
       c_src/tpuserve_driver.h c_src/tpuserve_driver.cc \
       c_src/tpuserve_model.h c_src/tpuserve_model.cc c_src/tpuserve.cc

$(ARTIFACT): $(SRCS)
	mkdir -p $(PRIV_DIR)
	$(CXX) $(CFLAGS) $(SRCS) -o $(ARTIFACT) $(LDFLAGS)
