#ifndef TPUSERVE_MODEL_H_
#define TPUSERVE_MODEL_H_

#include <vector>
#include <string>

#include "libtpu.h"
#include "tpuserve_nif_util.h"
#include "tpuserve_driver.h"

namespace tpuserve {

class TPUServeModel {
public:
  TPUServeModel(TPUServeDriver * driver,
                struct TpuCompiledProgramHandle* cph,
                std::vector<struct TpuBufferHandle*> input_buffer_handles,
                std::vector<struct TpuBufferHandle*> output_buffer_handles);

  ~TPUServeModel();

  // TODO: Status
  void Predict(std::vector<ErlNifBinary> &inputs, ErlNifBinary * output_buffer);

  size_t output_buffer_size(int i) const { return output_buffer_handles_.at(i)->size_in_bytes; }

  struct TpuCompiledProgramHandle* compiled_program_handle() const { return cph_; }

  std::vector<struct TpuBufferHandle*> input_buffer_handles() const { return input_buffer_handles_; }

  std::vector<struct TpuBufferHandle*> output_buffer_handles() const { return output_buffer_handles_; }

private:
  TPUServeDriver * driver_ = NULL;
  struct TpuCompiledProgramHandle* cph_;
  std::vector<struct TpuBufferHandle*> input_buffer_handles_;
  std::vector<struct TpuBufferHandle*> output_buffer_handles_;
  struct TpuLoadedProgramHandle* lph_;
  bool loaded_ = false;
  std::vector<TpuEvent*> execution_events_;
  int num_execution_events_ = 0;
};

TPUServeModel * CompileModel(TPUServeDriver * driver, std::string& model_path);

} // namespace tpuserve
#endif
