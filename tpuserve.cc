#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>

#include "logging.h"
#include "model.h"
#include "macro.h"
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

std::vector<std::unique_ptr<TPUServeModel>> CompileModelsInModelRepo(void* driver_fn,
                                                                     struct TpuDriver* driver,
                                                                     const char * model_repo) {
  std::string repo_dir(model_repo);
  for (const auto & entry : fs::directory_iterator(path))
    std::cout << entry.path() << std::endl;
}

int main(int argc, char ** argv) {
  // Load and Initialize TPU Driver
  LOG_INFO("Loading and initializing TPU Driver");
  struct TpuDriverFn driver_fn;
  void * handle = LoadAndInitializeDriver("libtpu.so", &driver_fn);

  LOG_INFO("Opening TPU Driver");
  struct TpuDriver* driver = driver_fn.TpuDriver_Open("local://");

  LOG_INFO("Querying System Information");
  struct TpuSystemInfo* info = driver_fn.TpuDriver_QuerySystemInfo(driver);
  driver_fn.TpuDriver_FreeSystemInfo(info);
  driver_fn.TpuDriver_Close(driver);

  // Close driver handle
  dlclose(handle);
}
