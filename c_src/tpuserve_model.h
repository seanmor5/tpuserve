#ifndef TPUSERVE_MODEL_H_
#define TPUSERVE_MODEL_H_

#include <vector>
#include <string>

#include "third_party/libtpu.h"

#include "tpuserve_nif_util.h"
#include "tpuserve_driver.h"

namespace tpuserve {

class TPUServeModel {
public:
  TPUServeModel(TPUServeDriver * driver,
                struct TpuCompiledProgramHandle* cph,
                std::vector<struct TpuBufferHandle*> input_buffer_handles,
                struct TpuBufferHandle * output_buffer_handle);

  ~TPUServeModel();

  // TODO: Status
  void Predict(std::vector<ErlNifBinary> &inputs, ErlNifBinary * output_buffer);

  size_t output_buffer_size() const { return output_buffer_handle_->size_in_bytes; }

  struct TpuCompiledProgramHandle* compiled_program_handle() const { return cph_; }

private:
  TPUServeDriver * driver_ = NULL;
  struct TpuCompiledProgramHandle* cph_;
  std::vector<struct TpuBufferHandle*> input_buffer_handles_;
  struct TpuBufferHandle * output_buffer_handle_;
  struct TpuLoadedProgramHandle * lph_;
  bool loaded_ = false;
};

} // namespace tpuserve
#endif
