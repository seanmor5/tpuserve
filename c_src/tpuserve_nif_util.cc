#include "tpuserve_nif_util.h"

namespace tpuserve {

namespace nif {

  // Status helpers

  ERL_NIF_TERM error(ErlNifEnv* env, const char* msg) {
    ERL_NIF_TERM atom = enif_make_atom(env, "error");
    ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
    return enif_make_tuple2(env, atom, msg_term);
  }

  ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term) {
    return enif_make_tuple2(env, ok(env), term);
  }

  ERL_NIF_TERM ok(ErlNifEnv* env) {
    return enif_make_atom(env, "ok");
  }

} // namespace nif

} // namespace tpuserve
