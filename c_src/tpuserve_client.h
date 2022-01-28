#ifndef TPUSERVE_CLIENT_H_
#define TPUSERVE_CLIENT_H_

#include "tpuserve_nif_util.h"
#include "tpuserve_driver.h"
#include "tpuserve_model.h"

// TPUServeClient is meant to be the public-facing API
// for working with TPUServe managed resources such as
// models and buffers.

namespace tpuserve {
namespace client {

  // Initializes the TPU driver given the path to the libtpu
  // shared object. The TPUServeDriver object manages access
  // to the TPU driver. It also ensures proper clean-up of
  // resources e.g. the dynamically opened shared library and
  // the TPU driver itself.
  TPUServeDriver * InitializeTpuDriver(std::string& shared_lib);

  // Loads the model at the given path. A model is a .hlo.txt
  // file which contains the program HLO. The TPUServeModel object
  // encapsulates the program, model input buffers, and model output
  // buffer.
  TPUServeModel * LoadModel(TPUServeDriver * driver, std::string& model_path);

  // Performs a prediction with the given model and inputs and returns
  // the result as an Erlang VM term.
  ERL_NIF_TERM Predict(ErlNifEnv * env,
                       TPUServeDriver * driver,
                       TPUServeModel * model,
                       std::vector<ErlNifBinary> inputs);

} // namespace client
} // namespace tpuserve
#endif