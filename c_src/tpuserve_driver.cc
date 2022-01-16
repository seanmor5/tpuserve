#include "tpuserve_driver.h"
#include "libtpu.h"
#include "logging.h"

namespace tpuserver {

  TPUServeDriver::~TPUServeDriver() {
    // Close driver
    if (driver_fn) {
      if (driver) {
        driver_fn.TpuDriver_Close(driver);
      }
    }

    // Close dlsym handle
    if (handle) {
      dlclose(handle);
    }
  }

  TPUServeDriver * GetTPUServeDriver(const char * shared_lib) {
    // Attempt to open libtpu.so
    void * handle;
    handle = dlopen(shared_lib, RTLD_NOW);
    if (NULL == handle) {
      LOG_FATAL("Error: %s", dlerror());
    }

    // Initialize driver
    struct TpuDriverFn driver_fn;
    PrototypeTpuDriver_Initialize* initialize_fn;
    *(void**)(&initialize_fn) = dlsym(handle, "TpuDriver_Initialize");
    initialize_fn(driver_fn, true);

    // Open driver
    struct TpuDriver * driver = driver_fn.TpuDriver_Open("local://");
    if (NULL == driver) {
      LOG_FATAL("Error: Failed to open driver");
    }

    // Raw pointers because the VM is annoying
    return new TPUServeDriver(handle, driver_fn, driver);
  }
}