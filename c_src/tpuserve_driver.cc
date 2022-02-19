#include <stdlib.h>
#include <dlfcn.h>

#include "third_party/libtpu.h"

#include "tpuserve_driver.h"
#include "logging.h"

namespace tpuserve {

  TPUServeDriver::~TPUServeDriver() {
    // Close driver
    if (driver_) {
      LOG_INFO("Closing TpuDriver");
      driver_fn_.TpuDriver_Close(driver_);
    }

    // Close dlsym handle
    if (handle_) {
      LOG_INFO("Closing libtpu.so handle");
      dlclose(handle_);
    }
  }
}
