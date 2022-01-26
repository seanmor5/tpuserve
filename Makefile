PRIV_DIR = priv

ARTIFACT = libtpuserve.so

CMAKE_FLAGS = -B$(PRIV_DIR)/build \
-DC_SRC=c_src -DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR)

all:
	mkdir -p $(PRIV_DIR)/build
	cmake $(CMAKE_FLAGS) .
	cd $(PRIV_DIR)/build && make
	mv $(PRIV_DIR)/build/$(ARTIFACT) $(PRIV_DIR)/$(ARTIFACT)