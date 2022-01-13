#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "logging.h"
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
  LOG_INFO("Loading and initializing TPU Driver");
  struct TpuDriverFn driver_fn;
  void * handle = LoadAndInitializeDriver("libtpu.so", &driver_fn);

  LOG_INFO("Opening TPU Driver");
  struct TpuDriver* driver = driver_fn.TpuDriver_Open("local://");

  LOG_INFO("Querying System Information");
  struct TpuSystemInfo* info = driver_fn.TpuDriver_QuerySystemInfo(driver);
  driver_fn.TpuDriver_FreeSystemInfo(info);

  // An example of simple program to sum two parameters.
  LOG_INFO("Compiling models in model directory");
  const char* model = "models/model.txt";
  FILE * fp = fopen(model, "r");
  fseek(fp, 0, SEEK_END);
  long int prog_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char * program = (char *) malloc(prog_size + 1);
  fread(program, 1, prog_size, fp);
  fclose(fp);

  TpuCompiledProgramHandle* cph = driver_fn.TpuDriver_CompileProgramFromText(driver, program, 1, 0, NULL);
  TpuLoadedProgramHandle* lph;

  if (NULL == cph) {
    LOG_FATAL("Unable to compile model");
  } else {
    TpuEvent* compile_events[] = { cph->event };
    lph = driver_fn.TpuDriver_LoadProgram(driver, 0, cph, 1, compile_events);
  }
  // Close driver handle
  dlclose(handle);
}
