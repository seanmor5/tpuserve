#include <string>

#include "third_party/libtpu.h"

#include "logging.h"
#include "tpuserve_model.h"
#include "tpuserve_driver.h"
#include "tpuserve_buffer.h"

namespace tpuserve {

TPUServeModel::TPUServeModel(
  TPUServeDriver * driver,
  TpuCompiledProgramHandle * cph,
  std::vector<std::unique_ptr<TPUServeBuffer>> input_buffers,
  std::unique_ptr<TPUServeBuffer> output_buffers
) : driver_(driver),
    cph_(cph),
    input_buffers_(std::move(input_buffers)),
    output_buffer_(std::move(output_buffers)) {
  if (NULL == cph) {
    LOG_ERROR("Program was not compiled successfully.");
    return;
  }

  TpuEvent * compile_events[] = { cph->event };
  lph_ =
    driver->driver_fn().TpuDriver_LoadProgram(
      driver->driver(), 0, cph, 1, compile_events
    );

  if (NULL == lph_) {
    LOG_ERROR("Program was not loaded successfully.");
    return;
  }

  // At this point we want to make sure the program is loaded
  // so we don't have to worry about racing between receiving
  // a request for an endpoint and the model still being loaded.
  TpuStatus * load_status =
    driver->driver_fn().TpuDriver_EventAwait(lph_->event, -1);
  if (load_status && load_status->code != 0) {
    LOG_ERROR("Program was not loaded successfully: %s", load_status->msg);
    return;
  }

  // If we've made it this far, the program was loaded! We
  // annotate that it was properly loaded so we know it
  // can be safely used during execution and so we know
  // we'll need to unload it on object destruction.
  loaded_ = true;
}

TPUServeModel::~TPUServeModel() {
  // If our model is loaded, unload it. At this point,
  // the server front-end needs to guarantee there are
  // no in-flight or actively being executed events.
  if (loaded_) {
    TpuEvent * unload_event =
      driver_->driver_fn().TpuDriver_UnloadProgram(
        driver_->driver(), lph_, 0, NULL
      );

    // There's no need to synchronize anything else with
    // the unload event, so we can just free the event
    // immediately after (checking of course to see if
    // we actually have an event to unload).
    if (unload_event) {
      driver_->driver_fn().TpuDriver_FreeEvent(unload_event);
    }
  }

  // Free the compiled program handle
  driver_->driver_fn().TpuDriver_FreeCompiledProgramHandle(cph_);
}

TpuEvent * TPUServeModel::Execute(int32_t wait_for_n, TpuEvent ** wait_for) {
  if (!loaded_) {
    // TODO: This should not be NULL
    return NULL;
  }

  std::vector<TpuBufferHandle *> input_tpu_handles;
  for (int i = 0; i < input_buffers_.size(); i++) {
    input_tpu_handles.push_back(input_buffer(i)->tpu_handle());
  }

  TpuBufferHandle * obh[] = { output_buffer_->tpu_handle() };

  DeviceAssignment da = {NULL, 0};

  TpuEvent * execute_event =
    driver_->driver_fn().TpuDriver_ExecuteProgram(
      driver_->driver(), lph_, input_tpu_handles.size(),
      input_tpu_handles.data(), 1, obh, da, wait_for_n, wait_for
    );

  return execute_event;
}

} // namespace tpuserve
