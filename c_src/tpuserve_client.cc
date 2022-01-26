#include <stdlib.h>
#include <iostream>

#include "third_party/libtpu.h"
#include "xla_data.pb.h"
#include "hlo.pb.h"

#include "tpuserve_client.h"
#include "tpuserve_model.h"
#include "logging.h"

namespace tpuserve {
namespace client {

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
  struct TpuAllocationShape output_alloc_shape = GetTpuAllocationShape(result_shape);
  struct TpuBufferHandle * output_handle =
    driver->driver_fn().TpuDriver_AllocateShape(
      driver->driver(), 0, 1, output_alloc_shape, 0, NULL
    );

  // TODO: Maybe this should be unique_ptr?
  return new TPUServeModel(driver, cph, std::move(input_handles), output_handle);
}

} // namespace client

} // namespace tpuserve