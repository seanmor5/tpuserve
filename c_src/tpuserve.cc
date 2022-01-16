#include <dlfcn.h>

#define INIT_SUCCESS 0
#define INIT_FAILURE -1

#include "tpuserve_nif_util.h"
#include "tpuserve_driver.h"
#include "libtpu.h"

static int open_resources(ErlNifEnv * env) {
  const char * mod = "TPUServe";

  int status = (
    tpuserve::nif::open_resource<tpuserve::TPUServeDriver>(env, mod, "TPUServeDriver")
  );

  return status ? INIT_SUCCESS : INIT_FAILURE;
}

static int load(ErlNifEnv * env, void ** priv, ERL_NIF_TERM load_info) {
  int status = open_resources(env);

  // Other work on NIF start-up

  return status;
}

ERL_NIF_TERM init_driver(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return tpuserve::nif::error(env, "Bad argument count.");
  }

  const char * shared_lib = "libtpu.so";

  // TODO: Status type
  TPUServeDriver * tpuserve_driver = GetTPUServeDriver(shared_lib);

  return tpuserve::nif::ok(env, tpuserve::nif::make<TPUServeDriver*>(env, tpuserve_driver));
}

static ErlNifFunc tpuserve_funcs[] = {
  {"init_driver", 0, init_driver, ERL_NIF_DIRTY_JOB_IO_BOUND}
};

ERL_NIF_INIT(Elixir.TPUServe.NIF, tpuserve_funcs, &load, NULL, NULL, NULL);