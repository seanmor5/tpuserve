#include <stdlib.h>
#include <dlfcn.h>
#include <fstream>
#include <sstream>

#include "third_party/libtpu.h"
#include "xla_data.pb.h"
#include "hlo.pb.h"

#include "tpuserve_nif_util.h"
#include "tpuserve_client.h"
#include "tpuserve_driver.h"
#include "tpuserve_model.h"
#include "tpuserve_buffer.h"
#include "logging.h"

namespace tpuserve {
namespace client {


TPUServeDriver * InitializeTpuDriver(std::string& shared_lib) {
  // Attempt to open libtpu.so
  void * handle;
  handle = dlopen(shared_lib.c_str(), RTLD_NOW);
  if (NULL == handle) {
    LOG_FATAL("Error: %s", dlerror());
  }

  // Initialize driver
  TpuDriverFn driver_fn;
  PrototypeTpuDriver_Initialize* initialize_fn;
  *(void**)(&initialize_fn) = dlsym(handle, "TpuDriver_Initialize");
  initialize_fn(&driver_fn, true);

  // Open driver
  TpuDriver * driver = driver_fn.TpuDriver_Open("local://");
  if (NULL == driver) {
    LOG_FATAL("Error: Failed to open driver");
  }

  // Raw pointers because the VM is annoying
  return new TPUServeDriver(handle, driver_fn, driver);
}

std::string ReadFile(std::string& file_path) {
  std::ifstream file_stream(file_path);

  if (!file_stream) {
    // TODO: This probably should not abort the entire
    // server if we can't read a file, so this should
    // probably be an std::optional or something instead.
    LOG_FATAL("Failed to read file from %s", file_path.c_str());
  }

  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  return buffer.str();
}

// TODO: StatusOr
TPUServeModel * LoadModel(TPUServeDriver * driver, std::string& model_path) {
  std::string model_text = ReadFile(model_path);

  TpuCompiledProgramHandle * cph =
    driver->driver_fn().TpuDriver_CompileProgramFromText(
      driver->driver(), model_text.c_str(), 1, 0, NULL
  );

  // TODO: Error check here

  // We can avoid depending on configurations by parsing the
  // program shape. It's not really clear from the libtpu
  // API that these are from the XLA data protobufs, but
  // if you do some digging it starts to make sense. The
  // program shape has the shapes of every parameter and
  // the result shape so we can allocate input buffers and
  // the single result buffer ahead of time. Input buffers
  // are in the order declared by the HLO program shape.
  // TODO: What to do if there's an explicit difference
  // between config shapes and program shape?
  xla::ProgramShapeProto program_shape;
  CompiledProgramShape * cph_shape =
    driver->driver_fn().TpuDriver_GetCompiledProgramShape(cph);
  program_shape.ParseFromArray(cph_shape->bytes, cph_shape->size);

  std::vector<std::unique_ptr<TPUServeBuffer>> input_buffers;
  input_buffers.reserve(program_shape.parameters_size());
  for (auto shape : program_shape.parameters()) {
    // Create a unique instance of one of this programs
    // parameters. The parameter buffers are owned by the
    // model. TPUServeBuffer's constructor takes care of
    // allocating the possibly nested buffer for us.
    input_buffers.push_back(std::make_unique<TPUServeBuffer>(driver, shape));
  }

  // Output will always be a single buffer (e.g. a tuple or array)
  // so we only ever need to allocate that single shape. Once again,
  // the TPUServeBuffer constructor will take care of proper allocation
  // according to the required buffer shape.
  xla::ShapeProto result_shape = program_shape.result();
  std::unique_ptr<TPUServeBuffer> output_buffer =
    std::make_unique<TPUServeBuffer>(driver, result_shape);
  // Ideally this would be a unique pointer explicitly owned by a
  // model manager object, but this object will be managed by the BEAM
  // and I haven't gotten resource pointers to work. So the usage of
  // new obligates us to explicitly delete the object once the last
  // reference to the object goes out of scope.
  return new TPUServeModel(driver, cph, std::move(input_buffers), std::move(output_buffer));
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
                                         const unsigned char * data,
                                         size_t data_size) {
  if (data_size != buffer->total_byte_size()) {
    LOG_ERROR(
      "Unable to copy data to device. \
        Data size %ld exceeded buffer size %ld",
      data_size, buffer->total_byte_size()
    );
    return {};
  }

  return buffer->PopulateBuffer(driver, data, data_size);
}

// Copies potentially nested tuple data from the device
// to a potentially nested Erlang VM tuple. All of the
// transfer events are completely synchronized before
// returning to the host.
ERL_NIF_TERM CopyDeviceToVM(ErlNifEnv * env,
                            TPUServeDriver * driver,
                            TPUServeBuffer * buffer,
                            int32_t wait_for_n,
                            TpuEvent ** wait_for) {
  return buffer->ToTerm(env, driver, wait_for_n, wait_for);
}

// TODO: Status
ERL_NIF_TERM Predict(ErlNifEnv * env,
                     TPUServeDriver * driver,
                     TPUServeModel * model,
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
  std::vector<TpuEvent*> transfer_events;
  for (int i = 0; i < model->number_of_inputs(); i++) {
    ErlNifBinary to_copy = inputs.at(i);
    std::vector<TpuEvent *> events_for_input =
      CopyHostToDevice(
        driver, model->input_buffer(i), const_cast<unsigned char *>(to_copy.data), to_copy.size
      );

    transfer_events.insert(
      transfer_events.end(), events_for_input.begin(), events_for_input.end()
    );
  }

  // Execute Model
  // TODO: I hate this pattern, just a bad design right now
  TpuEvent * execute_event = model->Execute(transfer_events.size(), transfer_events.data());

  // Transfer result to VM
  TpuEvent * wait_for_execution[] = { execute_event };
  ERL_NIF_TERM execution_result =
    CopyDeviceToVM(
      env, driver, model->output_buffer(), 1, wait_for_execution
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