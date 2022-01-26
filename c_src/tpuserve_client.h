#ifndef TPUSERVE_CLIENT_H_
#define TPUSERVE_CLIENT_H_

#include "tpuserve_driver.h"
#include "tpuserve_model.h"

namespace tpuserve {
namespace client {

  TPUServeModel * CompileModel(TPUServeDriver * driver, std::string& model_path);

} // namespace client
} // namespace tpuserve
#endif