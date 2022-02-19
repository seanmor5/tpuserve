#ifndef TPUSERVE_MODEL_H_
#define TPUSERVE_MODEL_H_

#include <vector>
#include <string>

#include "third_party/libtpu.h"

#include "tpuserve_buffer.h"
#include "tpuserve_driver.h"

namespace tpuserve {

class TPUServeModel {
public:
  TPUServeModel(TPUServeDriver * driver,
                TpuCompiledProgramHandle * cph,
                std::vector<std::unique_ptr<TPUServeBuffer>> input_buffers,
                std::unique_ptr<TPUServeBuffer> output_buffer);

  ~TPUServeModel();

  // Attempts to execute the underlying model program, assuming
  // the input buffers have been properly populated by an external
  // caller. Execution will be synchronized with the given events.
  TpuEvent * Execute(int32_t wait_for_n, TpuEvent ** wait_for);

  bool loaded() const { return loaded_; }

  int number_of_inputs() const { return input_buffers_.size(); }

  TPUServeBuffer * input_buffer(int i) const { return input_buffers_.at(i).get(); }

  TPUServeBuffer * output_buffer() const { return output_buffer_.get(); }

private:
  // Shared TPU driver for use during model loading/unloading
  // and execution. Ideally this would be a shared pointer
  // because ownership of the driver is expressed by multiple
  // objects, but unfortunately I can't get it to work nicely
  // with BEAM resource types.
  TPUServeDriver * driver_ = NULL;

  // Whether or not the underlying model program has been loaded.
  // This should be checked prior to model execution to ensure
  // we're not trying to do anything with an unloaded program.
  bool loaded_ = false;

  // Pointer to the underlying compiled TPU program. A properly
  // compiled program does not indicate that the model has been
  // properly loaded. Must be freed on object destruction.
  TpuCompiledProgramHandle* cph_ = NULL;

  // Pointer to the underlying loaded TPU program. Must be unloaded
  // on object destruction.
  TpuLoadedProgramHandle * lph_ = NULL;

  // Vector of unique pointers to input buffers. This model
  // has complete ownership of the input buffers. On model
  // destruction the input buffers are automatically deallocated.
  std::vector<std::unique_ptr<TPUServeBuffer>> input_buffers_;

  // Unique pointer to output buffer. Every model has a single
  // output (which could be a nested tuple or a regular buffer).
  // On model destruction the output buffer is automatically
  // deallocated.
  std::unique_ptr<TPUServeBuffer> output_buffer_;
};

} // namespace tpuserve
#endif
