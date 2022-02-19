#define INIT_SUCCESS 0
#define INIT_FAILURE -1

#include <string>

#include "third_party/libtpu.h"
#include "xla_data.pb.h"

#include "tpuserve_nif_util.h"
#include "tpuserve_driver.h"
#include "tpuserve_model.h"
#include "tpuserve_client.h"

void free_driver(ErlNifEnv * env, void * obj) {
  tpuserve::TPUServeDriver ** driver =
    reinterpret_cast<tpuserve::TPUServeDriver**>(obj);

  if (*driver != nullptr) {
    delete *driver;
    *driver = nullptr;
  }
}

void free_model(ErlNifEnv * env, void * obj) {
  tpuserve::TPUServeModel ** model =
    reinterpret_cast<tpuserve::TPUServeModel**>(obj);

  if (*model != nullptr) {
    delete *model;
    *model = nullptr;
  }
}

static int open_resources(ErlNifEnv * env) {
  const char * mod = "TPUServe";

  int status = (
    tpuserve::nif::open_resource<tpuserve::TPUServeDriver*>(env, mod, "TPUServeDriver", free_driver) &&
    tpuserve::nif::open_resource<tpuserve::TPUServeModel*>(env, mod, "TPUServeModel", free_model)
  );

  return status ? INIT_SUCCESS : INIT_FAILURE;
}

static int load(ErlNifEnv * env, void ** priv, ERL_NIF_TERM load_info) {
  int status = open_resources(env);

  // Other work on NIF start-up

  return status;
}

ERL_NIF_TERM
init_driver(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return tpuserve::nif::error(env, "Bad argument count.");
  }

  std::string shared_lib = "libtpu.so";

  // TODO: Status type
  tpuserve::TPUServeDriver * tpuserve_driver =
    tpuserve::client::InitializeTpuDriver(shared_lib);

  ERL_NIF_TERM driver_term =
    tpuserve::nif::make<tpuserve::TPUServeDriver*>(env, tpuserve_driver);

  return tpuserve::nif::ok(env, driver_term);
}

ERL_NIF_TERM
load_model(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return tpuserve::nif::error(env, "Bad argument count.");
  }

  std::string model_path;
  tpuserve::TPUServeDriver ** tpuserve_driver;

  if (!tpuserve::nif::get<tpuserve::TPUServeDriver*>(env, argv[0], tpuserve_driver)) {
    return tpuserve::nif::error(env, "Unable to get TPUServeDriver.");
  }
  if (!tpuserve::nif::get(env, argv[1], model_path)) {
    return tpuserve::nif::error(env, "Unable to get model path.");
  }

  tpuserve::TPUServeModel * tpuserve_model =
    tpuserve::client::LoadModel(*tpuserve_driver, model_path);

  ERL_NIF_TERM model_term =
    tpuserve::nif::make<tpuserve::TPUServeModel*>(env, tpuserve_model);

  return tpuserve::nif::ok(env, model_term);
}

ERL_NIF_TERM
predict(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return tpuserve::nif::error(env, "Bad argument count.");
  }

  tpuserve::TPUServeDriver ** tpuserve_driver;
  tpuserve::TPUServeModel ** tpuserve_model;
  std::vector<ErlNifBinary> inputs;

  if (!tpuserve::nif::get<tpuserve::TPUServeDriver*>(env, argv[0], tpuserve_driver)) {
    return tpuserve::nif::error(env, "Unable to get TPUServeDriver.");
  }
  if (!tpuserve::nif::get<tpuserve::TPUServeModel*>(env, argv[1], tpuserve_model)) {
    return tpuserve::nif::error(env, "Unable to get TPUServeModel.");
  }
  if (!tpuserve::nif::get_list(env, argv[2], inputs)) {
    return tpuserve::nif::error(env, "Unable to get inputs.");
  }

  ERL_NIF_TERM result_term =
    tpuserve::client::Predict(env, *tpuserve_driver, *tpuserve_model, inputs);

  return tpuserve::nif::ok(env, result_term);
}

static ErlNifFunc tpuserve_funcs[] = {
  {"init_driver", 0, init_driver, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"load_model", 2, load_model, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"predict", 3, predict, ERL_NIF_DIRTY_JOB_IO_BOUND}
};

ERL_NIF_INIT(Elixir.TPUServe.NIF, tpuserve_funcs, &load, NULL, NULL, NULL);
