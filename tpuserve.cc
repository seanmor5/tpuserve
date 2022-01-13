#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "libtpu.h"

void* LoadAndInitializeDriver(const char* shared_lib,
                              struct TpuDriverFn* driver_fn) {
  void* handle;
  handle = dlopen(shared_lib, RTLD_NOW);
  if (!handle) {
    LOG_FATAL("Error: %s", dlerror());
  }

  PrototypeTpuDriver_Initialize* initialize_fn;
  *(void**)(&initialize_fn) = dlsym(handle, "TpuDriver_Initialize");
  initialize_fn(driver_fn, true);

  return handle;
}

int main(int argc, char ** argv) {
  // Load and Initialize TPU Driver
  struct TpuDriverFn driver_fn;
  void * handle = LoadAndInitializeDriver("libtpu.so", &driver_fn);

  LOG_INFO("TPU Driver Version: %s", driver_fn.TpuDriver_Version());

  dlclose(handle);
}