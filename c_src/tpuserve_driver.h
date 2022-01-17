#ifndef TPUSERVE_DRIVER_H_
#define TPUSERVE_DRIVER_H_

#include <stdlib.h>

#include "libtpu.h"

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
                 struct TpuDriverFn driver_fn,
                 struct TpuDriver * driver) : handle_(handle),
                                              driver_fn_(driver_fn),
                                              driver_(driver) {}

  ~TPUServeDriver();


  struct TpuDriverFn driver_fn() const { return driver_fn_; }
  struct TpuDriver * driver() const { return driver_; }

private:
  // Pointer to libtpu.so dlsym handle
  void * handle_ = NULL;
  // TpuDriverFn Struct
  struct TpuDriverFn driver_fn_;
  // Pointer to TpuDriver
  struct TpuDriver * driver_ = NULL;
};

TPUServeDriver * GetTPUServeDriver(const char * shared_lib);
}
#endif
