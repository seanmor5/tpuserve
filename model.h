#ifndef TPUSERVE_MODEL_H_
#define TPUSERVE_MODEL_H_

#include "libtpu.h"
#include <vector>

namespace tpuserve {

class TPUServeModel {
public:
  TPUServeModel(struct TpuDriverFn driver_fn,
                struct TpuDriver* driver,
                struct TpuCompiledProgramHandle* cph,
                std::vector<struct TpuBufferHandle*> input_buffer_handles,
                std::vector<struct TpuBufferHandle*> output_buffer_handles)
                  : driver_fn_(driver_fn),
                    driver_(driver),
                    cph_(cph),
                    input_buffer_handles_(std::move(input_buffer_handles)),
                    output_buffer_handles_(std::move(output_buffer_handles)) {}

  ~TPUServeModel();

  struct TpuCompiledProgramHandle* compiled_program_handle() const { return cph_; }

  std::vector<struct TpuBufferHandle*> input_buffer_handles() const { return input_buffer_handles_; }

  std::vector<struct TpuBufferHandle*> output_buffer_handles() const { return output_buffer_handles_; }

private:
  // TODO: Is driver thread safe?
  struct TpuDriverFn driver_fn_;
  struct TpuDriver* driver_;
  struct TpuCompiledProgramHandle* cph_;
  std::vector<struct TpuBufferHandle*> input_buffer_handles_;
  std::vector<struct TpuBufferHandle*> output_buffer_handles_;
  struct TpuLoadedProgramHandle* lph_;
  bool loaded_ = false;
  struct TpuEvent** execution_events_ = NULL;
  int num_execution_events_ = 0;
};

}
#endif
