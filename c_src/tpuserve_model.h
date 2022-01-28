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
                struct TpuBufferHandle * output_buffer_handle,
                std::vector<struct TpuBufferHandle *>children);

  ~TPUServeModel();

  bool loaded() { return lph_ && loaded_; }

private:
  TPUServeDriver * driver_ = NULL;
  bool loaded_ = false;
  struct TpuCompiledProgramHandle* cph_ = NULL;
  struct TpuLoadedProgramHandle * lph_ = NULL;
  std::vector<std::unique_ptr<TPUServeBuffer>> input_buffer_handles_;
  std::unique_ptr<TPUServeBuffer> output_buffer_handle_;
};

} // namespace tpuserve
#endif
