#include <optional>
#include <queue>

#include "third_party/libtpu.h"

#include "tpuserve_buffer.h"
#include "tpuserve_nif_util.h"

namespace tpuserve {

// TODO: Shape specific logic?
struct TpuAllocationShape GetTpuAllocationShape(xla::ShapeProto shape) {
  struct TpuAllocationShape shape_;
  shape_.size = shape.ByteSizeLong();
  shape_.bytes = malloc(shape_.size);
  if (!shape.SerializeToArray(shape_.bytes, shape_.size)) {
    LOG_ERROR("Unable to serialize shape to array.");
    free(shape_.bytes);
    shape_.size = 0;
    shape_.bytes = nullptr;
  }
  return shape_;
}

// TODO: Shape specific logic?
bool IsTuple(xla::ShapeProto shape) {
  return shape.element_type() == xla::PrimitiveType::TUPLE;
}

struct BufferInternal AllocateBuffer(TPUServeDriver * driver, xla::ShapeProto shape) {
  struct TpuBufferHandle * root_buffer;
  size_t total_byte_size = 0;
  internal_buffers = std::nullopt;

  if (IsTuple(shape)) {
    size_t number_of_elements = shape.tuple_shapes_size();
    std::vector<struct BufferInternal> internal_buffers_value;
    std::vector<struct TpuBufferHandle *> internal_tpu_buffers;
    internal_buffers_value.reserve(number_of_elements);

    for (auto child_shape : shape.tuple_shapes()) {
      struct BufferInternal child_buffer = AllocateBuffer(child_shape);
      internal_buffers_value.push_back(child_buffer);
      internal_tpu_buffers.push_back(child_buffer.tpu_handle);
      total_byte_size += child_buffer->total_byte_size;
    }

    root_buffer =
      driver->driver_fn().TpuDriver_AllocateTuple(
        driver->driver(), 0, 1, internal_tpu_buffers.size(),
        internal_tpu_buffers.data(), 0, NULL
      );

    internal_buffers = std::make_optional<std::vector<BufferInternal>>(internal_buffers_value);
  } else {
    struct TpuAllocationShape alloc_shape = GetTpuAllocationShape(shape);
    root_buffer =
      driver->driver_fn().TpuDriver_AllocateShape(
        driver->driver(), 0, 1, alloc_shape, 0, NULL
      );
    total_byte_size = root_buffer->size_in_bytes;
  }

  return { .tpu_handle=root_buffer, .internal_buffers=internal_buffers, .total_byte_size=total_byte_size };

}

TPUServeBuffer::TPUServeBuffer(TPUServeDriver * driver, xla::ShapeProto shape) {
  driver_ = driver;
  internal_buffer_handle_ = AllocateBuffer(driver, shape);
}

TPUServeBuffer::~TPUServeBuffer() {
  // TODO: I'm assuming this will free all child memory
  // as well, but that's a dangerous game to play.
  // TODO: I think we can guarantee that at the time of
  // destruction the model won't have any in-flight
  // requests, but I will need to draw this one out.
  TpuEvent * dealloc_event =
    driver_->driver_fn().TpuDriver_Deallocate(
      driver_->driver(), internal_buffer_handle_->tpu_handle, 0, NULL
    );

  if (dealloc_event) {
    driver_->driver_fn().TpuDriver_FreeEvent(dealloc_event);
  }
}

} // namespace tpuserve