#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <stdarg.h>

/** Basic Logger
 *
 *  Provides macros for logging events that take place on the server
 *  at different log levels. These are basically glog style macros,
 *  except they don't provide a timestamp of the event. The API can
 *  be easily extended to support timestamps.
 *
 *  The contents of msg are NOT sanitized, so none of these methods
 *  should __EVER__ log user input.
 *
 *  https://stackoverflow.com/questions/1644868/define-macro-for-debug-printing-in-c
 *  https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
 */

#define LOG_INFO(msg, ...) fprintf(stdout, "[INFO] %s():%s:%d " msg "\n", \
                            __func__, __FILE__, __LINE__, ##__VA_ARGS__);

#define LOG_WARN(msg, ...) fprintf(stdout, "[WARN] %s():%s:%d " msg "\n", \
                            __func__, __FILE__, __LINE__, ##__VA_ARGS__);

#define LOG_ERROR(msg, ...) fprintf(stderr, "[ERROR] %s():%s:%d " msg "\n", \
                            __func__, __FILE__, __LINE__, ##__VA_ARGS__);

#define LOG_FATAL(msg, ...) fprintf(stderr, "[FATAL] %s():%s:%d " msg "\n", \
                            __func__, __FILE__, __LINE__, ##__VA_ARGS__);   \
                            exit(EXIT_FAILURE);

#endif
// end of file
