#ifndef TPUSERVE_MACRO_H_
#define TPUSERVE_MACRO_H_

#include "logging.h"

// Macro to be used to consume possibly NULL values.
// Will bind the LHS if not NULL, otherwise abort
#define ASSIGN_OR_RETURN_ON_NULL(lhs, rexpr)             \
  ASSIGN_OR_RETURN_ON_NULL_IMPL(                         \
    CONCAT_NAME(                                         \
      _status_or_value, __COUNTER__),                    \
  lhs, rexpr)

#define ASSIGN_OR_RETURN_NULL_IMPL(statusor, lhs, rexpr) \
  auto val = (rexpr);                                    \
  if (NULL == val) {                                     \
    LOG_FATAL("Unexpected NULL value");                  \
  }                                                      \
  lhs = val;

#endif