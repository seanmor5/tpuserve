#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <memory>
#include <iostream>
#include <regex>
#include <vector>

#include "logging.h"
#include "model.h"
#include "macro.h"
#include "libtpu.h"

// TODO: Make drivers shared pointers

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

// Replace with C++ way to do this
// Memory must be freed by caller
char * ReadFileToBuffer(const char * path) {
  FILE * fp = fopen(path, "r");
  fseek(fp, 0, SEEK_END);
  size_t fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char * buffer = (char *) malloc(sizeof(char) * fsize + 1);
  fread(buffer, sizeof(char), fsize, fp);
  fclose(fp);
  return buffer;
}

std::vector<std::unique_ptr<tpuserve::TPUServeModel>> CompileModelsInModelRepo(struct TpuDriverFn driver_fn,
                              struct TpuDriver* driver,
                              const char * model_repo) {
  std::filesystem::path repo_path(model_repo);
  std::regex model_pattern(".*\\.hlo.txt$");
  std::vector<std::unique_ptr<tpuserve::TPUServeModel>> models;

  for (const auto & entry : std::filesystem::directory_iterator(repo_path)) {
    const auto fname = entry.path().string();
    if (std::regex_match(fname, model_pattern)) {
      LOG_INFO("Found model entry %s", entry.path().string().c_str());
      char * prog = ReadFileToBuffer(entry.path().string().c_str());
      LOG_INFO("Compiling %s...", entry.path().string().c_str());
      struct TpuCompiledProgramHandle* cph =
        driver_fn.TpuDriver_CompileProgramFromText(driver, prog, 1, 0, NULL);
      if (cph) {
        LOG_INFO("Successfully compiled %s", entry.path().string().c_str());
        std::vector<TpuBufferHandle*> in_handle({});
        std::vector<TpuBufferHandle*> out_handle({});
        models.push_back(std::make_unique<tpuserve::TPUServeModel>(driver_fn, driver, cph, std::move(in_handle), std::move(out_handle)));
      }
      free(prog);
    }
  }
  return std::move(models);
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

  // Attempt to compile and load models in model directory
  CompileModelsInModelRepo(driver_fn, driver, "models");

  // Close driver handle and driver
  driver_fn.TpuDriver_Close(driver);
  dlclose(handle);
}
