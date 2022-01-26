#include <string>

#include "third_party/libtpu.h"

#include "logging.h"
#include "tpuserve_nif_util.h"
#include "tpuserve_model.h"
#include "tpuserve_driver.h"

namespace tpuserve {

  TPUServeModel::TPUServeModel(TPUServeDriver * driver,
                               struct TpuCompiledProgramHandle * cph,
                               std::vector<struct TpuBufferHandle *> input_buffer_handles,
                               struct TpuBufferHandle * output_buffer_handle) : driver_(driver),
                                                                                cph_(cph),
                                                                                input_buffer_handles_(std::move(input_buffer_handles)),
                                                                                output_buffer_handle_(output_buffer_handle) {
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

    // Deallocate all input buffers
    for (auto buf : input_buffer_handles_) {
      struct TpuEvent* dealloc_event =
        driver_->driver_fn().TpuDriver_Deallocate(driver_->driver(), buf, 0, NULL);

      if (dealloc_event) {
        driver_->driver_fn().TpuDriver_FreeEvent(dealloc_event);
      }
    }

    // Deallocate output buffer
    struct TpuEvent* dealloc_event =
      driver_->driver_fn().TpuDriver_Deallocate(driver_->driver(), output_buffer_handle_, 0, NULL);

    if (dealloc_event) {
      driver_->driver_fn().TpuDriver_FreeEvent(dealloc_event);
    }
  }

  // TODO: Status
  // TODO: Multiple outputs
  // Assumes output buffer is allocated properly
  void TPUServeModel::Predict(std::vector<ErlNifBinary> &inputs, ErlNifBinary * output_buffer) {
    if (!loaded_ || lph_ == NULL) {
      LOG_ERROR("Inference Error: Model was not properly loaded");
      return;
    }

    // Populate input buffers
    std::vector<struct TpuEvent*> transfer_events;
    transfer_events.reserve(input_buffer_handles_.size());
    for (int i = 0; i < input_buffer_handles_.size(); i++) {
      // TODO: Cleanup on another thread?
      struct TpuBufferHandle * inp_handle = input_buffer_handles_.at(i);
      struct TpuEvent * allocate_event[] = { inp_handle->event };
      struct TpuEvent * transfer_event =
        driver_->driver_fn().TpuDriver_TransferToDevice(
          driver_->driver(), inputs.at(i).data, inp_handle, 1, allocate_event
        );

      transfer_events.push_back(transfer_event);
    }

    // Execute Model
    DeviceAssignment device_assignment = {NULL, 0};
    struct TpuBufferHandle * out_handles[] = { output_buffer_handle_ };
    struct TpuEvent * execution_event =
      driver_->driver_fn().TpuDriver_ExecuteProgram(
        driver_->driver(), lph_, input_buffer_handles_.size(),
        input_buffer_handles_.data(), 1,
        out_handles, device_assignment, transfer_events.size(),
        transfer_events.data()
      );

    // Transfer from device
    struct TpuEvent * execution_events[] = { execution_event };
    struct TpuEvent * transfer_event =
      driver_->driver_fn().TpuDriver_TransferFromDevice(
        driver_->driver(), output_buffer_handle_, (*output_buffer).data, 1, execution_events
      );

    TpuStatus * status = driver_->driver_fn().TpuDriver_EventAwait(transfer_event, 10000000);

    if (status && status->code != 0) {
      LOG_ERROR("Something went wrong in execution: %s", status->msg);
    }

    // Clean up
    // TODO: Free every event
    return;
  }
} // namespace tpuserve
