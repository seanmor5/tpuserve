#include <string>

#include "libtpu.h"
#include "tpuserve_model.h"
#include "tpuserve_driver.h"

namespace tpuserve {

  TPUServeModel::~TPUServeModel() {
    // If our model is loaded, unload it
    if (loaded_) {
      struct TpuEvent* unload_event =
        driver_->driver_fn().TpuDriver_UnloadProgram(driver_->driver(), lph_, num_execution_events_, execution_events_);

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

  // TODO: Move this function somewhere else
  // TODO: StatusOr
  TPUServeModel * CompileModel(TPUServeDriver * driver, std::string& model_path) {
    FILE * fp = fopen(model_path.c_str(), "r");
    fseek(fp, 0, SEEK_END);
    size_t prog_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char * model_text = (char *) malloc(sizeof(char) * prog_size + 1);
    fread(model_text, sizeof(char), prog_size, fp);
    model_text[prog_size] = '\0';

    struct TpuCompiledProgramHandle * cph =
      driver->driver_fn().TpuDriver_CompileProgramFromText(driver->driver(), model_text, 1, 0, NULL);

    return new TPUServeModel(driver, cph, {}, {});
  }
} // namespace tpuserve
