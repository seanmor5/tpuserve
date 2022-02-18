#include <optional>
#include <queue>

#include "third_party/libtpu.h"

#include "tpuserve_nif_util.h"
#include "tpuserve_buffer.h"
#include "logging.h"

namespace tpuserve {

// TODO: Shape specific logic?
TpuAllocationShape GetTpuAllocationShape(xla::ShapeProto shape) {
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

BufferInternal AllocateBuffer(TPUServeDriver * driver, xla::ShapeProto shape) {
  TpuBufferHandle * root_buffer;
  size_t total_byte_size = 0;
  std::optional<std::vector<BufferInternal>> internal_buffers;

  if (IsTuple(shape)) {
    size_t number_of_elements = shape.tuple_shapes_size();
    std::vector<BufferInternal> internal_buffers_value;
    std::vector<TpuBufferHandle *> internal_tpu_buffers;
    internal_buffers_value.reserve(number_of_elements);

    for (auto child_shape : shape.tuple_shapes()) {
      BufferInternal child_buffer = AllocateBuffer(driver, child_shape);
      internal_buffers_value.push_back(child_buffer);
      internal_tpu_buffers.push_back(child_buffer.tpu_handle);
      total_byte_size += child_buffer.total_byte_size;
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
    internal_buffers = std::nullopt;
  }

  BufferInternal internal_handle = {
    .shape=shape,
    .tpu_handle=root_buffer,
    .children=internal_buffers,
    .total_byte_size=total_byte_size
  };

  return internal_handle;
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
      driver_->driver(), internal_buffer_handle_.tpu_handle, 0, NULL
    );

  if (dealloc_event) {
    driver_->driver_fn().TpuDriver_FreeEvent(dealloc_event);
  }
}

std::vector<TpuEvent *> TPUServeBuffer::PopulateBuffer(TPUServeDriver * driver,
                                                       const unsigned char * data,
                                                       size_t data_size) {
  size_t total_data_copied = 0;
  std::queue<BufferInternal*> to_populate;
  std::vector<TpuEvent *> transfer_events;
  to_populate.push(&internal_buffer_handle_);

  while (total_data_copied < data_size && !to_populate.empty()) {
    BufferInternal * populating = to_populate.front();
    to_populate.pop();

    if (populating->children.has_value()) {
      for (auto child : populating->children.value()) {
        to_populate.push(&child);
      }
    } else {
      size_t size_to_copy = populating->tpu_handle->size_in_bytes;
      TpuEvent * allocate_event[] = { populating->tpu_handle->event };

      TpuEvent * transfer_event =
        driver->driver_fn().TpuDriver_TransferToDevice(
          driver->driver(), &data[total_data_copied], populating->tpu_handle, 1, allocate_event
        );

      transfer_events.push_back(transfer_event);
      total_data_copied += size_to_copy;
    }
  }

  return transfer_events;
}

TpuStatus * CopyDeviceToHostInternal(TPUServeDriver * driver,
                                     BufferInternal internal,
                                     char * dst,
                                     int32_t wait_for_n,
                                     TpuEvent ** wait_for) {
  TpuEvent * transfer_event =
    driver->driver_fn().TpuDriver_TransferFromDevice(
      driver->driver(), internal.tpu_handle, dst,
      wait_for_n, wait_for
    );

  // TODO: Timeout option
  TpuStatus * transfer_status =
    driver->driver_fn().TpuDriver_EventAwait(transfer_event, -1);

  return transfer_status;
}

ERL_NIF_TERM CopyDeviceToVMInternal(ErlNifEnv * env,
                                    TPUServeDriver * driver,
                                    BufferInternal internal,
                                    int32_t wait_for_n,
                                    TpuEvent ** wait_for) {
  if (internal.children.has_value()) {
    std::vector<ERL_NIF_TERM> inner_terms;
    inner_terms.reserve(internal.children.value().size());

    for (auto child : internal.children.value()) {
      ERL_NIF_TERM child_term =
        CopyDeviceToVMInternal(env, driver, child, wait_for_n, wait_for);
      inner_terms.push_back(child_term);
    }

    return enif_make_tuple_from_array(env, inner_terms.data(), inner_terms.size());
  } else {
    size_t size_of_buffer = internal.tpu_handle->size_in_bytes;
    ErlNifBinary binary;
    enif_alloc_binary(size_of_buffer, &binary);

    TpuStatus * transfer_status =
      CopyDeviceToHostInternal(driver, internal, reinterpret_cast<char *>(binary.data), wait_for_n, wait_for);

    if (transfer_status && transfer_status->code != 0) {
      LOG_ERROR("Something went wrong in transfer: %s", transfer_status->msg);
      return nif::error(env, "error");
    } else {
      return nif::make(env, binary);
    }
  }
}

ERL_NIF_TERM TPUServeBuffer::ToTerm(ErlNifEnv * env,
                                    TPUServeDriver * driver,
                                    int32_t wait_for_n,
                                    TpuEvent ** wait_for) {
  return CopyDeviceToVMInternal(env, driver, internal_buffer_handle_, wait_for_n, wait_for);
}

} // namespace tpuserve