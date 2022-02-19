#include "third_party/libtpu.h"
#include "xla_data.pb.h"

#include "logging.h"
#include "shape.h"

namespace tpuserve {

namespace shape {

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

  int64_t ByteSizeOfShape(const xla::ShapeProto& shape) {
    int64_t prim_size = ByteSizeOfPrimitiveType(shape.element_type());
    if (shape.dimensions_size() == 0) {
      // scalar case
      return prim_size;
    }
    int64_t elems = 1;
    for (auto dim : shape.dimensions()) {
      elems *= dim;
    }

    return elems * prim_size;
  }

  int64_t ByteSizeOfPrimitiveType(xla::PrimitiveType primitive_type) {
    switch (primitive_type) {
      case xla::PrimitiveType::PRED:
        return sizeof(int8_t);
      case xla::PrimitiveType::S8:
        return sizeof(int8_t);
      case xla::PrimitiveType::S16:
        return sizeof(int16_t);
      case xla::PrimitiveType::S32:
        return sizeof(int32_t);
      case xla::PrimitiveType::S64:
        return sizeof(int64_t);
      case xla::PrimitiveType::U8:
        return sizeof(uint8_t);
      case xla::PrimitiveType::U16:
        return sizeof(uint16_t);
      case xla::PrimitiveType::U32:
        return sizeof(uint32_t);
      case xla::PrimitiveType::U64:
        return sizeof(uint64_t);
      case xla::PrimitiveType::BF16:
        return sizeof(float) / 2;
      case xla::PrimitiveType::F16:
        return sizeof(float) / 2;
      case xla::PrimitiveType::F32:
        return sizeof(float);
      case xla::PrimitiveType::F64:
        return sizeof(double);
      case xla::PrimitiveType::TOKEN:
        return 0;
      case xla::PrimitiveType::TUPLE:
      case xla::PrimitiveType::OPAQUE_TYPE:
        LOG_FATAL("Primitive type has no definitive size");
      default:
        LOG_FATAL("Unhandled primitive type");
    }
  }

  bool IsTuple(const xla::ShapeProto& shape) {
    return shape.element_type() == xla::PrimitiveType::TUPLE;
  }

  bool LayoutsAreEqual(const xla::LayoutProto& lhs, const xla::LayoutProto& rhs) {
    if (lhs.minor_to_major_size() != rhs.minor_to_major_size()) {
      return false;
    }

    bool are_equal = true;
    for (int i = 0; i < lhs.minor_to_major_size(); i++) {
      are_equal = are_equal && (lhs.minor_to_major(i) == rhs.minor_to_major(i));
    }

    return are_equal;
  }

  int64_t MultidimensionalIndexToLinearIndex(const xla::ShapeProto& shape,
                                             std::vector<int64_t> multi_index) {
    int64_t scale = 1;
    int64_t linear_index = 0;
    bool first = true;
    for (auto dimension : shape.layout().minor_to_major()) {
      if (first) {
        // Avoid two multiplies on the first loop iteration
        linear_index = multi_index[dimension];
        scale = shape.dimensions(dimension);
        first = false;
      } else {
        linear_index += scale * multi_index[dimension];
        scale *= shape.dimensions(dimension);
      }
    }
    return linear_index;
  }

  bool BumpIndices(const xla::ShapeProto& shape, std::vector<int64_t>& indices) {
    for (int64_t dimno = indices.size() - 1; dimno >= 0; --dimno) {
      int64_t limit = shape.dimensions(dimno);
      if (indices[dimno] + 1 < limit) {
        indices[dimno]++;
        // Whenever an index of a dimension is increased, it means that all
        // following dimensions have maxed out, so they must go to 0.
        std::fill(indices.begin() + dimno + 1, indices.end(), 0);
        return true;
      }
    }
    return false;
  }

  xla::ShapeProto MakeRowMajor(const xla::ShapeProto& shape) {
    xla::ShapeProto row_major_shape;
    row_major_shape.set_element_type(shape.element_type());

    for (auto dim : shape.dimensions()) {
      row_major_shape.add_dimensions(dim);
    }

    xla::LayoutProto row_major_layout;
    for (int64_t i = shape.dimensions_size() - 1; i >= 0; i--) {
      row_major_layout.add_minor_to_major(i);
    }
    xla::LayoutProto * mutable_layout = row_major_shape.mutable_layout();
    *mutable_layout = row_major_layout;

    return row_major_shape;
  }
}

}