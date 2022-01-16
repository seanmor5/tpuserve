CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -O3 -Wall -Wextra \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
	 -shared -std=c++17

LDFLAGS = -ldl

ARTIFACT = $(PRIV_DIR)/libtpuserve.so

$(ARTIFACT): c_src/tpuserve.cc c_src/model.cc c_src/model.h
	mkdir -p $(PRIV_DIR)
	$(CXX) $(CFLAGS) c_src/tpuserve.cc c_src/model.cc c_src/model.h -o $(ARTIFACT)  $(LDFLAGS)
