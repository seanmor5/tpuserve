#ifndef TPUSERVE_NIF_UTIL_H_
#define TPUSERVE_NIF_UTIL_H_

#include "erl_nif.h"

namespace tpuserve {

namespace nif {

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

} // namespace nif

} // namespace tpuserve

#endif