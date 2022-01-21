#include <string>

#include "libtpu.h"
#include "logging.h"
#include "tpuserve_nif_util.h"
#include "tpuserve_model.h"
#include "tpuserve_driver.h"

namespace tpuserve {

  TPUServeModel::TPUServeModel(TPUServeDriver * driver,
                               TpuCompiledProgramHandle * cph,
                               std::vector<TpuBufferHandle*> input_buffer_handles,
                               std::vector<TpuBufferHandle*> output_buffer_handles) : driver_(driver),
                                                                                      cph_(cph),
                                                                                      input_buffer_handles_(std::move(input_buffer_handles)),
                                                                                      output_buffer_handles_(std::move(output_buffer_handles)) {
  if (NULL == cph) {
    LOG_ERROR("Program was not compiled successfully.");
    return;
  }

  TpuEvent * compile_events[] = { cph->event };
  lph_ = driver->driver_fn().TpuDriver_LoadProgram(driver->driver(), 0, cph, 1, compile_events);
}

  TPUServeModel::~TPUServeModel() {
    // If our model is loaded, unload it
    if (loaded_) {
      struct TpuEvent* unload_event =
        driver_->driver_fn().TpuDriver_UnloadProgram(driver_->driver(), lph_, num_execution_events_, &execution_events_[0]);

      if (unload_event) {
        driver_->driver_fn().TpuDriver_FreeEvent(unload_event);
      }
    }

    // TODO: Free all execution events

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

    // Deallocate all output buffers
    for (auto buf : output_buffer_handles_) {
      struct TpuEvent* dealloc_event =
        driver_->driver_fn().TpuDriver_Deallocate(driver_->driver(), buf, 0, NULL);

      if (dealloc_event) {
        driver_->driver_fn().TpuDriver_FreeEvent(dealloc_event);
      }
    }
  }

  // TODO: Status
  // TODO: Multiple outputs
  // Assumes output buffer is allocated properly
  void TPUServeModel::Predict(std::vector<ErlNifBinary> &inputs, ErlNifBinary * output_buffer) {
    if (NULL == lph_) {
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
    struct TpuEvent * execution_event =
      driver_->driver_fn().TpuDriver_ExecuteProgram(
        driver_->driver(), lph_, input_buffer_handles_.size(),
        &input_buffer_handles_[0], output_buffer_handles_.size(),
        &output_buffer_handles_[0], device_assignment, transfer_events.size(),
        &transfer_events[0]
      );

    // Transfer from device
    struct TpuBufferHandle * obh = output_buffer_handles_.at(0);
    struct TpuEvent * execution_events[] = { execution_event };
    struct TpuEvent * transfer_event =
      driver_->driver_fn().TpuDriver_TransferFromDevice(
        driver_->driver(), obh, (*output_buffer).data, 1, execution_events
      );

    TpuStatus * status = driver_->driver_fn().TpuDriver_EventAwait(transfer_event, 10000000);

    LOG_INFO("%d", status->code);
    LOG_INFO("%s", status->msg);

    // Clean up
    execution_events_.push_back(execution_event);
    // TODO: Free every event
    return;
  }

  // TODO: Move this function elsewhere
  struct TpuAllocationShape GetTpuAllocationShape(size_t size) {
    TpuAllocationShape shape_;
    shape_.size = size;
    shape_.bytes = malloc(shape_.size);
    return shape_;
  }

  // TODO: Move this function somewhere else
  // TODO: StatusOr
  TPUServeModel * CompileModel(TPUServeDriver * driver,
                               std::string& model_path,
                               std::vector<int> input_sizes,
                               std::vector<int> output_sizes) {
    FILE * fp = fopen(model_path.c_str(), "r");
    fseek(fp, 0, SEEK_END);
    size_t prog_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char * model_text = (char *) malloc(sizeof(char) * prog_size + 1);
    fread(model_text, sizeof(char), prog_size, fp);
    model_text[prog_size] = '\0';

    struct TpuCompiledProgramHandle * cph =
      driver->driver_fn().TpuDriver_CompileProgramFromText(driver->driver(), model_text, 1, 0, NULL);

    std::vector<struct TpuBufferHandle*> input_handles;
    input_handles.reserve(input_sizes.size());
    for (auto size : input_sizes) {
      struct TpuAllocationShape shape = GetTpuAllocationShape(size);
      struct TpuBufferHandle * input_handle =
        driver->driver_fn().TpuDriver_AllocateShape(driver->driver(), 0, 1, shape, 0, NULL);
      input_handles.push_back(input_handle);
    }

    std::vector<struct TpuBufferHandle*> output_handles;
    output_handles.reserve(output_sizes.size());
    for (auto size : output_sizes) {
      struct TpuBufferHandle * output_handle =
        driver->driver_fn().TpuDriver_Allocate(driver->driver(), 0, 1, size, 0, NULL);
      output_handles.push_back(output_handle);
    }

    return new TPUServeModel(driver, cph, std::move(input_handles), std::move(output_handles));
  }
} // namespace tpuserve
