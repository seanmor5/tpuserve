#ifndef TPUSERVE_DRIVER_H_
#define TPUSERVE_DRIVER_H_

#include <stdlib.h>

#include "third_party/libtpu.h"

// Class which holds a pointer to the driver, driver_fn,
// and handler. On construction, this class:
//
//    1) Attempts to open `libtpu.so`
//    2) Initializes TpuDriverFn
//    3) Opens TpuDriver
//
// On destruction, this class:
//
//    1) Closes the TpuDriver
//    2) Closes `libtpu.so` handler
//
// For now we hold a single reference to this object in a
// persistent term. AFAIK for a single TPU vx-8 that's fine,
// I'm not really sure how that changes when working with
// a TPU Pod.
//
// TODO: Is this object thread safe?

namespace tpuserve {

class TPUServeDriver {
public:

  TPUServeDriver(void * handle,
                 TpuDriverFn driver_fn,
                 TpuDriver * driver) : handle_(handle),
                                              driver_fn_(driver_fn),
                                              driver_(driver) {}

  ~TPUServeDriver();


  TpuDriverFn driver_fn() const { return driver_fn_; }
  TpuDriver * driver() const { return driver_; }

private:
  // Pointer to libtpu.so dlsym handle
  void * handle_ = NULL;
  // TpuDriverFn
  TpuDriverFn driver_fn_;
  // Pointer to TpuDriver
  TpuDriver * driver_ = NULL;
};
}
#endif
