#include <stdlib.h>
#include <iostream>

#include "third_party/libtpu.h"
#include "xla_data.pb.h"
#include "hlo.pb.h"

#include "tpuserve_client.h"
#include "tpuserve_model.h"
#include "tpuserve_buffer.h"
#include "tpuserve_driver.h"
#include "logging.h"

namespace tpuserve {
namespace client {

// TODO: StatusOr
TPUServeModel * CompileModel(TPUServeDriver * driver, std::string& model_path) {
  // TODO: More C++-isms here
  FILE * fp = fopen(model_path.c_str(), "r");
  fseek(fp, 0, SEEK_END);
  size_t prog_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char * model_text = (char *) malloc(sizeof(char) * prog_size + 1);
  fread(model_text, sizeof(char), prog_size, fp);
  model_text[prog_size] = '\0';

  struct TpuCompiledProgramHandle * cph =
    driver->driver_fn().TpuDriver_CompileProgramFromText(
      driver->driver(), model_text, 1, 0, NULL
    );

  // TODO: Error check here

  // We can avoid depending on configurations by parsing the
  // program shape. It's not really clear from the libtpu
  // API that these are from the XLA data protobufs, but
  // if you do some digging it starts to make sense. The
  // program shape has the shapes of every parameter and
  // the result shape so we can allocate input buffers and
  // the single result buffer ahead of time.
  xla::ProgramShapeProto program_shape_proto;
  struct CompiledProgramShape * cph_shape =
    driver->driver_fn().TpuDriver_GetCompiledProgramShape(cph);
  program_shape_proto.ParseFromArray(cph_shape->bytes, cph_shape->size);

  std::vector<struct TpuBufferHandle*> input_handles;
  input_handles.reserve(program_shape_proto.parameters_size());
  for (auto shape : program_shape_proto.parameters()) {
    // Convert parameter shape proto to TPU allocation shape
    // which can be allocated with driver function
    struct TpuAllocationShape alloc_shape = GetTpuAllocationShape(shape);
    // Allocate shape of parameter and return a handle to
    // the underlying shape so we can track where data
    // needs to be when an inference request comes in
    struct TpuBufferHandle * input_handle =
      driver->driver_fn().TpuDriver_AllocateShape(
        driver->driver(), 0, 1, alloc_shape, 0, NULL
      );
    input_handles.push_back(input_handle);
  }

  // Output will always be a single buffer (e.g. a tuple or array)
  // so we only ever need to allocate that single shape. The buffer
  // will need to be decomposed later on for it to make sense
  xla::ShapeProto result_shape = program_shape_proto.result();
  struct TpuBufferHandle * output_handle;
  std::vector<struct TpuBufferHandle *> children;

  if (IsTuple(result_shape)) {
    // TODO: This will end up being a recursive thing for nested
    // tuples, so we need to move this out of here and abstract
    // all of it into a single buffer construct.
    children.reserve(result_shape.tuple_shapes_size());
    for (auto child_shape : result_shape.tuple_shapes()) {
      struct TpuAllocationShape child_tpu_shape = GetTpuAllocationShape(child_shape);
      struct TpuBufferHandle * child_buffer =
        driver->driver_fn().TpuDriver_AllocateShape(
          driver->driver(), 0, 1, child_tpu_shape, 0, NULL
        );
      children.push_back(child_buffer);
    }

    output_handle =
      driver->driver_fn().TpuDriver_AllocateTuple(
        driver->driver(), 0, 1, children.size(), children.data(), 0, NULL
      );

  } else {
    struct TpuAllocationShape output_alloc_shape = GetTpuAllocationShape(result_shape);
    output_handle =
      driver->driver_fn().TpuDriver_AllocateShape(
        driver->driver(), 0, 1, output_alloc_shape, 0, NULL
      );
  }

  // TODO: Maybe this should be unique_ptr?
  return new TPUServeModel(driver, cph, std::move(input_handles), output_handle, std::move(children));
}



std::vector<TpuEvent *> CopyHostToBufferInternal(TPUServerDriver * driver,
                                                 struct BufferInternal internal,
                                                 char * data,
                                                 size_t data_size) {
  size_t total_data_copied = 0;
  std::queue<BufferInternal> to_populate;
  std::vector<TpuEvent *> transfer_events;
  to_populate.push(internal);

  while (total_data_copied < data_size && !to_populate.empty()) {
    struct BufferInternal populating = to_populate.pop();

    if (populating.children.has_value()) {
      for (auto child : populating.children.value()) {
        to_populate.push(child);
      }
    } else {
      size_t size_to_copy = populating.tpu_handle->size_in_bytes;
      TpuEvent * allocate_event[] = { populating.tpu_handle->event };

      char * src = &(data[total_data_copied]);
      TpuEvent * transfer_event =
        driver->driver_fn().TpuDriver_TransferToDevice(
          driver->driver(), src, populating.tpu_handle, 1, allocate_event
        );

      transfer_events.push_back(transfer_event);
      total_data_copied += size_to_copy;
    }
  }

  return std::move(transfer_events);
}

// Copies flat buffer from host to device by populating
// child buffers in a DFS-order. For example, if a tuple
// consisted of buffers:
//
//    {B1, {B2, B3, {B4, B5}, B6, {B7, B8}}}
//
// The buffers would get populated in order from B1 - B8.
// If the size of the data given by data_size is not equal
// to the total size of the buffer, then the copy will
// fail and no data will be transferred. If data_size exceeds
// the actual size of data, it is undefined behavior.
std::vector<TpuEvent *> CopyHostToDevice(TPUServeDriver * driver,
                                         TPUServeBuffer * buffer,
                                         char * data,
                                         size_t data_size) {
  if (data_size != total_byte_size()) {
    LOG_ERROR(
      "Unable to copy data to device. \
        Data size %d exceeded buffer size %d",
      data_size, total_byte_size()
    );
    return nullptr;
  }

  return CopyHostToBufferInternal(driver, buffer->buffer_handle(), data, data_size);
}

TpuStatus * CopyDeviceToHostInternal(TPUServeDriver * driver,
                                     struct BufferInternal internal,
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
        CopyDeviceToVMInternal(driver, child, wait_for_n, wait_for);
      inner_terms.push_back(child_term);
    }

    return enif_make_tuple_from_array(env, inner_terms.data(), inner_terms.size());
  } else {
    size_t size_of_buffer = internal.tpu_handle->size_in_bytes;
    ErlNifBinary binary;
    enif_alloc_binary(size_of_buffer, &binary);

    TpuStatus * transfer_status =
      CopyDeviceToHostInternal(driver, internal, wait_for_n, wait_for);

    if (transfer_status && transfer_status->code != 0) {
      LOG_ERROR("Something went wrong in transfer: %s", transfer_status->msg);
      return nif::atom(env, "error");
    } else {
      return nif::make(env, binary);
    }
  }
}

// Copies potentially nested tuple data from the device
// to a potentially nested Erlang VM tuple. All of the
// transfer events are completely synchronized before
// returning to the host.
ERL_NIF_TERM CopyDeviceToVM(TPUServeDriver * driver,
                            TPUServeBuffer * buffer,
                            ErlNifEnv * env,
                            int32_t wait_for_n,
                            TpuEvent ** wait_for) {
  return CopyDeviceToVMInternal(env, driver, buffer->buffer_handle(), wait_for_n, wait_for);
}

// TODO: Status
// TODO: Multiple outputs
// Assumes output buffer is allocated properly
ERL_NIF_TERM Predict(TPUServeDriver * driver,
                     TPUServeModel * model,
                     ErlNifEnv * env,
                     std::vector<ErlNifBinary> inputs) {
  if (!model->loaded()) {
    LOG_ERROR("Inference Error: Model was not properly loaded");
    return nif::error(env, "Inference error");
  }

  if (inputs.size() != model->number_of_inputs()) {
    LOG_ERROR("Inference Error: Number of model input buffers does \
              not match inputs given.");
  }
  // Populate input buffers
  std::vector<struct TpuEvent*> transfer_events;
  for (int i = 0; i < model->number_of_inputs(); i++) {
    ErlNifBinary to_copy = inputs.at(i);
    std::vector<TpuEvent *> events_for_input =
      CopyHostToDevice(
        driver, model->buffer(i), inputs.data, inputs.size
      );

    transfer_events.insert(
      transfer_events.end(), events_for_input.begin(), events_for_input.end()
    );
  }

  // Execute Model
  // TODO: I hate this pattern, just a bad design right now
  TpuEvent * execute_event = model->Execute(transfer_events.size(), transfer_events.data());

  // Transfer from device
  TpuEvent * execution_events[] = { execute_event };

  ERL_NIF_TERM execution_result =
    CopyDeviceToVM(
      driver, model->output_buffer(), env, 1, execution_events
    );

  // Clean up
  for (auto event : transfer_events) {
    driver->driver_fn().TpuDriver_FreeEvent(event);
  }
  driver->driver_fn().TpuDriver_FreeEvent(execute_event);

  // Return
  return execution_result;
}

} // namespace client

} // namespace tpuserve