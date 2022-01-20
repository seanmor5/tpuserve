#ifndef TPUSERVE_NIF_UTIL_H_
#define TPUSERVE_NIF_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "erl_nif.h"

namespace tpuserve {

namespace nif {

// Status helpers

// Helper for returning `{:error, msg}` from NIF.
ERL_NIF_TERM error(ErlNifEnv* env, const char* msg);

// Helper for returning `{:ok, term}` from NIF.
ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term);

// Helper for returning `:ok` from NIF.
ERL_NIF_TERM ok(ErlNifEnv* env);

int get(ErlNifEnv* env, ERL_NIF_TERM term, int *var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var);

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int> &var);

int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<ErlNifBinary> &var);

// Template struct for resources. The struct lets us use templates
// to store and retrieve open resources later on. This implementation
// is the same as the approach taken in the goertzenator/nifpp
// C++11 wrapper around the Erlang NIF API.
template <typename T>
struct resource_object {
  static ErlNifResourceType *type;
};
template<typename T> ErlNifResourceType* resource_object<T>::type = 0;

// Default destructor passed when opening a resource. The default
// behavior is to invoke the underlying objects destructor and
// set the resource pointer to NULL.
template <typename T>
void default_dtor(ErlNifEnv* env, void * obj) {
  T* resource = reinterpret_cast<T*>(obj);
  resource->~T();
  resource = nullptr;
}

// Opens a resource for the given template type T. If no
// destructor is given, uses the default destructor defined
// above.
template <typename T>
int open_resource(ErlNifEnv* env,
                  const char* mod,
                  const char* name,
                  ErlNifResourceDtor* dtor = nullptr) {
  if (dtor == nullptr) {
    dtor = &default_dtor<T>;
  }
  ErlNifResourceType *type;
  ErlNifResourceFlags flags = ErlNifResourceFlags(ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER);
  type = enif_open_resource_type(env, mod, name, dtor, flags, NULL);
  if (type == NULL) {
    resource_object<T>::type = 0;
    return -1;
  } else {
    resource_object<T>::type = type;
  }
  return 1;
}
// Returns a resource of the given template type T.
template <typename T>
ERL_NIF_TERM get(ErlNifEnv* env, ERL_NIF_TERM term, T* &var) {
  return enif_get_resource(env, term,
                           resource_object<T>::type,
                           reinterpret_cast<void**>(&var));
}

ERL_NIF_TERM make(ErlNifEnv * env, ErlNifBinary binary);

// Creates a reference to the given resource of type T. We
// use the move constructor by default because some XLA
// objects delete the copy-constructor. The move is intended
// to represent a transfer of ownership of the object to
// the VM.
template <typename T>
ERL_NIF_TERM make(ErlNifEnv* env, T &var) {
  void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
  new(ptr) T(std::move(var));
  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  enif_release_resource(ptr);
  return ret;
}

} // namespace nif

} // namespace tpuserve

#endif
