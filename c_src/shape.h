#ifndef SHAPE_H_
#define SHAPE_H_

#include "third_party/libtpu.h"
#include "xla_data.pb.h"

namespace tpuserve {

namespace shape {

  TpuAllocationShape GetTpuAllocationShape(xla::ShapeProto shape);

  int64_t ByteSizeOfShape(const xla::ShapeProto& shape);

  int64_t ByteSizeOfPrimitiveType(xla::PrimitiveType ptype);

  bool IsTuple(const xla::ShapeProto& shape);

  bool LayoutsAreEqual(const xla::LayoutProto& lhs, const xla::LayoutProto& rhs);

  int64_t MultidimensionalIndexToLinearIndex(const xla::ShapeProto& shape, std::vector<int64_t> multi_index);

  bool BumpIndices(const xla::ShapeProto& shape, std::vector<int64_t>& multi_index);

  xla::ShapeProto MakeRowMajor(const xla::ShapeProto& shape);

} // namespace shape
} // namespace tpuserve

#endif