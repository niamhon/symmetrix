#pragma once
#include <pybind11/numpy.h>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

// Shared Python input checks keep CPU and CUDA graph conventions identical.
namespace lsc_binding {
namespace py = pybind11;
using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline std::vector<int> indices(const py::array& input, const char* name)
{
    const auto kind = input.dtype().kind();
    if (kind != 'i' and kind != 'u')
        throw std::invalid_argument(std::string("MACELSC: ") + name + " must have an integer dtype");
    auto values = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(input);
    if (not values) throw std::invalid_argument(std::string("MACELSC: cannot read ") + name);
    std::vector<int> result(values.size());
    for (py::ssize_t i = 0; i < values.size(); ++i) {
        const auto value = values.data()[i];
        if (value < 0 or value > std::numeric_limits<int>::max())
            throw std::invalid_argument(std::string("MACELSC: ") + name + " index is out of range");
        result[i] = static_cast<int>(value);
    }
    return result;
}

inline RealArray real_array(const py::array& input, const char* name)
{
    const auto kind = input.dtype().kind();
    if (kind != 'f' and kind != 'i' and kind != 'u')
        throw std::invalid_argument(std::string("MACELSC: ") + name + " must have a real numeric dtype");
    auto result = RealArray::ensure(input);
    if (not result) throw std::invalid_argument(std::string("MACELSC: cannot read ") + name);
    return result;
}

struct GraphInput {
    std::vector<int> types, edges;
    RealArray vectors;
    py::ssize_t count;
    std::span<const int> senders() const { return {edges.data(), size_t(count)}; }
    std::span<const int> receivers() const {
        return {count > 0 ? edges.data() + count : edges.data(), size_t(count)};
    }
};

inline GraphInput graph_input(const py::array& types, const py::array& edges, const py::array& vectors)
{
    if (types.ndim() != 1 or edges.ndim() != 2 or edges.shape(0) != 2
        or vectors.ndim() != 2 or vectors.shape(1) != 3 or vectors.shape(0) != edges.shape(1))
        throw std::invalid_argument("MACELSC: expected node_types[N], edge_index[2,E], edge_vectors[E,3]");
    if (types.size() > std::numeric_limits<int>::max() or edges.shape(1) > std::numeric_limits<int>::max())
        throw std::invalid_argument("MACELSC: excessive node or edge count");
    return {indices(types, "node_types"), indices(edges, "edge_index"), real_array(vectors, "edge_vectors"), edges.shape(1)};
}

} // namespace lsc_binding
