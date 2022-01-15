#include "libtpu.h"
#include "model.h"
#include "macro.h"

namespace tpuserve {

  TPUServeModel::~TPUServeModel() {
    // If our model is loaded, unload it
    if (loaded_) {
      struct TpuEvent* unload_event =
        driver_fn_.TpuDriver_UnloadProgram(driver_, lph_, num_execution_events_, execution_events_);

      if (unload_event) {
        driver_fn_.TpuDriver_FreeEvent(unload_event);
      }
    }

    // TODO: Free all execution events

    // Free the compiled program handle
    driver_fn_.TpuDriver_FreeCompiledProgramHandle(cph_);

    // Deallocate all input buffers
    for (auto buf : input_buffer_handles_) {
      struct TpuEvent* dealloc_event =
        driver_fn_.TpuDriver_Deallocate(driver_, buf, 0, NULL);

      if (dealloc_event) {
        driver_fn_.TpuDriver_FreeEvent(dealloc_event);
      }
    }

    // Deallocate all output buffers
    for (auto buf : output_buffer_handles_) {
      struct TpuEvent* dealloc_event =
        driver_fn_.TpuDriver_Deallocate(driver_, buf, 0, NULL);

      if (dealloc_event) {
        driver_fn_.TpuDriver_FreeEvent(dealloc_event);
      }
    }
  }
}
