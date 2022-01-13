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
  struct TpuDriver* driver = driver_fn.TpuDriver_Open("local://");

  LOG_INFO("TPU Driver Version: %s", driver_fn.TpuDriver_Version());

  // Open *.hlo in model directory
  const char * model = "models/model.hlo";
  FILE * fp = fopen(model, "r");
  fseek(fp, 0, SEEK_END);
  long int prog_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  program = malloc(prog_size + 1);
  fread(buffer, 1, prog_size, f);
  program[prog_size] = '\0';

  // Compile HLO
  struct TpuCompiledProgramHandle* cph =
    driver_fn.TpuDriver_CompileProgramFromText(driver, program, 1, 0, NULL);

  TpuEvent* compile_events[] = {cph->event};

  // Close driver handle
  dlclose(handle);
}