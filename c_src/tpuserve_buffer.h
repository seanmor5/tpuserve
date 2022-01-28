#ifndef TPUSERVE_BUFFER_H_
#define TPUSERVE_BUFFER_H_

#include <optional>

#include "third_party/libtpu.h"
#include "xla_data.pb.h"

#include "tpuserve_driver.h"

namespace tpuserve {

// This class is essentially a wrapper around TpuBufferHandle
// which provides functions that are shape agnostic. This
// is particularly useful when working with Tuple input and
// outputs. On construction this class will correctly allocate
// a TpuBufferHandle, as well as any child buffers in the case
// of tuple shapes. On destruction, this class will free the
// allocated buffer memory.

struct BufferInternal;

struct BufferInternal {
  // At the top level, there will always be a reference to
  // some underlying TPU memory. Even in the case of a tuple,
  // we will have this reference to pass to execution or
  // elsewhere.
  struct TpuBufferHandle * tpu_handle;
  // If the underlying shape of the data structure is a tuple,
  // then this will contain references to the BufferInternal
  // structs which make up the tuple. This is recursive because
  // you can have a tuple of tuples or regular flat buffers.
  std::optional<std::vector<BufferInternal>> internal_buffers;
  // Represents the total byte size of all buffers contained
  // within the root buffer.
  size_t total_byte_size;
}

class TPUServeBuffer {
public:

  TPUServeBuffer(TPUServeDriver * driver, xla::ShapeProto shape);

  ~TPUServeBuffer();

  size_t root_buffer_size() const {
    return internal_buffer_handle_->tpu_handle->size_in_bytes;
  }

  size_t total_byte_size() const {
    return internal_buffer_handle_->total_byte_size;
  }

  void CopyHostToDevice(char * data, size_t data_size);

  ERL_NIF_TERM TPUServeBuffer::CopyDeviceToVM(ErlNifEnv * env,
                                              int32_t wait_for_n,
                                              TpuEvent ** wait_for);

  TpuEvent ** TPUServeBuffer::CopyHostToDevice(char * data, size_t data_size);

private:
  TPUServeDriver * driver_;
  struct BufferInternal internal_buffer_handle_;
}

} // namespace tpuserve


#endif