#include <string>

#include "third_party/libtpu.h"

#include "logging.h"
#include "tpuserve_nif_util.h"
#include "tpuserve_model.h"
#include "tpuserve_driver.h"
#include "tpuserve_buffer.h"

namespace tpuserve {

  TPUServeModel::TPUServeModel(TPUServeDriver * driver,
                               struct TpuCompiledProgramHandle * cph,
                               std::vector<std::unique_ptr<TPUServeBuffer>> input_buffer_handles,
                               std::unique_ptr<TPUServeBuffer> output_buffer_handle) : driver_(driver),
                                                                                       cph_(cph),
                                                                                       input_buffer_handles_(std::move(input_buffer_handles)),
                                                                                       output_buffer_handle_(std::move(output_buffer_handle)) {
  if (NULL == cph) {
    LOG_ERROR("Program was not compiled successfully.");
    return;
  }

  TpuEvent * compile_events[] = { cph->event };
  lph_ = driver->driver_fn().TpuDriver_LoadProgram(driver->driver(), 0, cph, 1, compile_events);

  if (NULL == lph_) {
    LOG_ERROR("Program was not loaded successfully.");
    return;
  }

  // At this point we want to make sure the program is loaded
  // so we don't have to worry about racing between receiving
  // a request for an endpoint and the model still being loaded.
  TpuStatus * load_status = driver->driver_fn().TpuDriver_EventAwait(lph_->event, -1);
  if (load_status && load_status->code != 0) {
    LOG_ERROR("Program was not loaded successfully: %s", load_status->msg);
    return;
  }

  // If we've made it this far, the program was loaded! We
  // annotate that it was properly loaded so we know to
  // unload it on destruction
  loaded_ = true;
}

  TPUServeModel::~TPUServeModel() {
    // If our model is loaded, unload it
    if (loaded_) {
      struct TpuEvent* unload_event =
        driver_->driver_fn().TpuDriver_UnloadProgram(driver_->driver(), lph_, 0, NULL);

      if (unload_event) {
        driver_->driver_fn().TpuDriver_FreeEvent(unload_event);
      }
    }

    // Free the compiled program handle
    driver_->driver_fn().TpuDriver_FreeCompiledProgramHandle(cph_);
  }
} // namespace tpuserve
