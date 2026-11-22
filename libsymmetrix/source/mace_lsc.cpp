#include "mace_lsc.hpp"
#include "mace_lsc_forward.hpp"
#include "mace_lsc_charge_exchange.hpp"
#include "mace_lsc_reverse_exchange.hpp"
#include <optional>
#include "mace_lsc_model.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <stdexcept>

#include "mace_polar.hpp"
#include "nlohmann/json.hpp"
#include "sphericart.hpp"

namespace {
using json = nlohmann::json;
using Matrix = std::vector<std::vector<double>>;
constexpr int C = 128;
const std::vector<int> hidden = {C, C};
const std::vector<int> target = {C, C, C, C};

[[noreturn]] void fail(const std::string& path, const std::string& message)
{
    throw std::invalid_argument("MACELSC: " + path + ": " + message);
}

void require(bool condition, const std::string& path, const std::string& message)
{
    if (not condition) fail(path, message);
}

const json& field(const json& j, const std::string& key, const std::string& path = "")
{
    const auto name = path.empty() ? key : path + "." + key;
    require(j.is_object() and j.contains(key), name, "missing required field");
    return j.at(key);
}

int length(const json& j, const std::string& path)
{
    require(j.is_array() and j.size() <= std::numeric_limits<int>::max(), path, "expected an array");
    return static_cast<int>(j.size());
}

int integer(const json& j, const std::string& path, int minimum = 0,
            int maximum = std::numeric_limits<int>::max())
{
    require(j.is_number_integer(), path, "expected an integer");
    if (j.is_number_unsigned()) {
        const auto value = j.get<std::uint64_t>();
        require(value <= static_cast<std::uint64_t>(maximum), path, "integer is out of range");
        require(value >= static_cast<std::uint64_t>(minimum), path, "integer is out of range");
        return static_cast<int>(value);
    }
    const auto value = j.get<std::int64_t>();
    require(value >= minimum and value <= maximum, path, "integer is out of range");
    return static_cast<int>(value);
}

bool finite_value(double value)
{
    // The project enables -ffast-math. Inspect bits so validation cannot be
    // optimized away by its assumption that all floating-point inputs are finite.
    const auto bits = std::bit_cast<std::uint64_t>(value);
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

double number(const json& j, const std::string& path, bool positive = false)
{
    require(j.is_number(), path, "expected a finite number");
    const double value = j.get<double>();
    require(finite_value(value), path, "expected a finite number");
    require(not positive or value > 0, path, "expected a positive number");
    return value;
}

std::string text(const json& j, const std::string& path)
{
    require(j.is_string(), path, "expected a string");
    return j.get<std::string>();
}

void expect(const json& j, const std::string& path, const std::string& expected)
{
    require(text(j, path) == expected, path, "expected " + expected);
}

bool boolean(const json& j, const std::string& path)
{
    require(j.is_boolean(), path, "expected a boolean");
    return j.get<bool>();
}

std::vector<int> integers(const json& j, const std::string& path, int size)
{
    require(length(j, path) == size, path, "incorrect array length");
    std::vector<int> out;
    for (int i = 0; i < size; ++i) out.push_back(integer(j[i], path + "[" + std::to_string(i) + "]"));
    return out;
}

std::vector<double> numbers(const json& j, const std::string& path, int size)
{
    require(length(j, path) == size, path, "incorrect array length");
    std::vector<double> out;
    out.reserve(size);
    for (int i = 0; i < size; ++i) out.push_back(number(j[i], path + "[" + std::to_string(i) + "]"));
    return out;
}

Matrix matrix(const json& j, const std::string& path, int rows, int columns)
{
    require(length(j, path) == rows, path, "incorrect number of rows");
    Matrix out;
    for (int i = 0; i < rows; ++i) out.push_back(numbers(j[i], path + "[" + std::to_string(i) + "]", columns));
    return out;
}

std::vector<int> offsets(const std::vector<int>& mul)
{
    std::vector<int> out = {0};
    for (int l = 0; l < mul.size(); ++l) out.push_back(out.back() + (2 * l + 1) * mul[l]);
    return out;
}

MACEPolar::Linear linear(const json& j, const std::string& path,
                         const std::vector<int>& in, const std::vector<int>& out)
{
    MACEPolar::Linear result;
    result.in = integers(field(j, "in", path), path + ".in", in.size());
    result.out = integers(field(j, "out", path), path + ".out", out.size());
    require(result.in == in and result.out == out, path, "incompatible linear-map dimensions");
    result.off_in = offsets(in);
    result.off_out = offsets(out);
    const auto& weights = field(j, "weights", path);
    require(length(weights, path + ".weights") == out.size(), path + ".weights", "incorrect l blocks");
    for (int l = 0; l < out.size(); ++l) {
        result.weights.push_back(numbers(weights[l], path + ".weights[" + std::to_string(l) + "]", in[l] * out[l]));
        result.weights_trans.push_back(std::vector<double>(in[l] * out[l]));
        for (int a = 0; a < in[l]; ++a)
            for (int b = 0; b < out[l]; ++b)
                result.weights_trans[l][b * in[l] + a] = result.weights[l][a * out[l] + b];
    }
    result.bias = numbers(field(j, "bias", path), path + ".bias", out[0]);
    return result;
}

struct Oxidation {
    std::array<double, 2> range;
    double factor;
    MACEPolar::Linear linear;
};

Oxidation oxidation(const json& j, const std::string& path)
{
    expect(field(j, "basis", path), path + ".basis", "ChebychevBasis");
    require(integer(field(j, "num_basis", path), path + ".num_basis") == 8, path + ".num_basis", "expected eight functions");
    const auto bounds = numbers(field(j, "range", path), path + ".range", 2);
    require(bounds[0] < bounds[1], path + ".range", "expected increasing bounds");
    return {{bounds[0], bounds[1]}, number(field(j, "factor", path), path + ".factor", true),
            linear(field(j, "linear", path), path + ".linear", {8}, {C})};
}

struct Convolution {
    std::vector<int> in, out;
    int num_weights;
    std::vector<std::array<int, 4>> spec;
    std::vector<double> coefficients;
};

Convolution convolution(const json& j, const std::string& path,
                        const std::vector<int>& in, const std::vector<int>& out, int weights)
{
    require(integer(field(j, "num_channels", path), path + ".num_channels") == C, path, "expected 128 channels");
    Convolution result;
    result.in = integers(field(j, "in", path), path + ".in", in.size());
    result.out = integers(field(j, "out", path), path + ".out", out.size());
    result.num_weights = integer(field(j, "num_weights", path), path + ".num_weights");
    require(result.in == in and result.out == out and result.num_weights == weights, path, "incompatible convolution dimensions");
    const auto& spec = field(j, "spec", path);
    const int count = length(spec, path + ".spec");
    require(count > 0, path + ".spec", "expected convolution terms");
    for (int i = 0; i < count; ++i) {
        const auto name = path + ".spec[" + std::to_string(i) + "]";
        const auto term = integers(spec[i], name, 4);
        require(term[0] <= offsets(out).back() - C and term[1] <= offsets(in).back() - C
                and term[2] < 16 and term[3] <= weights - C
                and term[0] % C == 0 and term[1] % C == 0 and term[3] % C == 0,
                name, "invalid channelwise convolution indices");
        result.spec.push_back({term[0], term[1], term[2], term[3]});
    }
    result.coefficients = numbers(field(j, "coefficients", path), path + ".coefficients", count);
    return result;
}

struct Radial {
    int functions, knots, outputs;
    Matrix weights;
    std::unique_ptr<CubicSplineSet<double>> splines;
    std::unique_ptr<CubicSplineSet<double>> projected_splines;
};

Radial radial(const json& values, const json& derivatives, const json& projection,
              const std::string& path, int knots, int outputs, double r_min, double r_cut)
{
    const int functions = length(values, path + ".values");
    require(functions > 0, path + ".values", "expected radial functions");
    auto v = matrix(values, path + ".values", functions, knots);
    auto d = matrix(derivatives, path + ".derivs", functions, knots);
    auto w = matrix(projection, path + ".weights", functions, outputs);
    return {functions, knots, outputs, std::move(w),
            std::make_unique<CubicSplineSet<double>>(r_min, r_cut, std::move(v), std::move(d)), nullptr};
}

struct Product {
    std::vector<std::array<int, 3>> basis_spec, output_spec;
    std::vector<double> coefficients, weights;
};

Product product(const json& file, int layer, int elements)
{
    const auto B = "B" + std::to_string(layer), M = "M" + std::to_string(layer);
    const auto& spec = field(file, B + "_spec");
    Product result;
    std::set<int> defined;
    for (int i = 0; i < 16; ++i) defined.insert(i);
    const int count = length(spec, B + "_spec");
    for (int i = 0; i < count; ++i) {
        const auto name = B + "_spec[" + std::to_string(i) + "]";
        const auto term = integers(spec[i], name, 3);
        require(term[0] >= 16 and term[0] - 16 < count and term[1] < term[0] and term[2] < term[0]
                and defined.contains(term[1]) and defined.contains(term[2]), name, "invalid product dependency indices");
        result.basis_spec.push_back({term[0], term[1], term[2]});
        defined.insert(term[0]);
    }
    result.coefficients = numbers(field(file, B + "_coefficients"), B + "_coefficients", count);
    const auto& outputs = field(file, M + "_spec");
    const int terms = length(outputs, M + "_spec");
    require(terms > 0, M + "_spec", "expected product outputs");
    for (int i = 0; i < terms; ++i) {
        const auto name = M + "_spec[" + std::to_string(i) + "]";
        const auto term = integers(outputs[i], name, 3);
        require((term[0] == 0 and term[2] == 1) or (term[0] == 1 and term[2] == 3), name, "incompatible scalar/vector product output");
        for (int m = 0; m < term[2]; ++m)
            require(term[1] <= std::numeric_limits<int>::max() - m and defined.contains(term[1] + m), name, "undefined product basis value");
        result.output_spec.push_back({term[0], term[1], term[2]});
    }
    const auto& weights = field(file, M + "_weights");
    require(length(weights, M + "_weights") == elements, M + "_weights", "incorrect species count");
    for (int e = 0; e < elements; ++e)
        for (const auto& row : matrix(weights[e], M + "_weights[" + std::to_string(e) + "]", terms, C))
            result.weights.insert(result.weights.end(), row.begin(), row.end());
    return result;
}

struct Layer {
    MACEPolar::Linear up, output, product_linear, readout;
    std::vector<MACEPolar::Linear> skip;
    Convolution conv;
    Radial radial;
    Product product;
    std::vector<double> up_features;
};

struct Source {
    Oxidation embedding;
    MACEPolar::Modulation modulation;
    MACEPolar::Linear linear1, linear2, readout, multipoles;
    Convolution conv;
    Radial radial;
    std::vector<double> sender_features, receiver_features;
    double transfer_scale;
};

std::vector<double> oxidation_embeddings(const Oxidation& embedding, const std::vector<double>& formal)
{
    const int elements = formal.size();
    std::vector<double> basis(long(elements) * 8), result(long(elements) * C);
    for (int e = 0; e < elements; ++e) {
        const double x = (formal[e] - embedding.range[0]) * embedding.factor - 1.0;
        double previous = 1.0, current = x;
        for (int k = 0; k < 8; ++k) {
            basis[e * 8 + k] = current;
            const double next = 2.0 * x * current - previous;
            previous = current; current = next;
        }
    }
    embedding.linear.forward(elements, basis.data(), 8, result.data(), C, false);
    return result;
}

// Polar's CPU evaluator folds the radial projection into spline tables. Prepare
// that representation lazily, once, retaining the compact export for later GPU work.
void prepare_radial(Radial& radial, double r_min, double r_cut)
{
    if (radial.projected_splines) return;
    Matrix hidden(radial.functions, std::vector<double>(radial.knots));
    Matrix hidden_derivs = hidden;
    std::vector<double> values(radial.functions), derivs(radial.functions);
    const double h = (r_cut - r_min) / (radial.knots - 1);
    for (int k = 0; k < radial.knots; ++k) {
        const double r = k == radial.knots - 1 ? r_cut : r_min + k * h;
        radial.splines->evaluate_derivs(r, values, derivs);
        for (int f = 0; f < radial.functions; ++f) {
            hidden[f][k] = values[f];
            hidden_derivs[f][k] = derivs[f];
        }
    }
    Matrix projected(radial.outputs, std::vector<double>(radial.knots, 0.0));
    Matrix projected_derivs = projected;
    #pragma omp parallel for
    for (int o = 0; o < radial.outputs; ++o)
        for (int f = 0; f < radial.functions; ++f)
            for (int k = 0; k < radial.knots; ++k) {
                projected[o][k] += radial.weights[f][o] * hidden[f][k];
                projected_derivs[o][k] += radial.weights[f][o] * hidden_derivs[f][k];
            }
    radial.projected_splines = std::make_unique<CubicSplineSet<double>>(
        r_min, r_cut, std::move(projected), std::move(projected_derivs));
}

void compute_layer(Layer& layer, std::span<const int> node_types,
                   std::span<const int> senders, const std::vector<int>& starts,
                   const std::vector<int>& edges, const std::vector<double>& radii,
                   const std::vector<double>& Y, const std::vector<double>& input,
                   std::vector<double>& A, std::vector<double>& M, std::vector<double>& H, int input_nodes=-1)
{
    const int n = node_types.size();
    const int input_width = layer.up.off_in.back();
    const int up_width = layer.up.off_out.back();
    const int message_width = layer.output.off_in.back();
    if(input_nodes<0)input_nodes=n;
    layer.up_features.resize(long(input_nodes) * up_width);
    auto& up = layer.up_features;
    layer.up.forward(input_nodes, input.data(), input_width, up.data(), up_width, false);
    std::vector<double> message(long(n) * message_width, 0.0);
    // Group by receiver so each thread owns a message row; preserve the input
    // order within each row and avoid races/atomic additions on shared atoms.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        std::vector<double> R(layer.radial.outputs);
        auto out = message.data() + long(i) * message_width;
        for (int position = starts[i]; position < starts[i + 1]; ++position) {
            const int edge = edges[position];
            layer.radial.projected_splines->evaluate(radii[edge], R);
            const auto x = up.data() + long(senders[edge]) * up_width;
            for (int t = 0; t < layer.conv.spec.size(); ++t) {
                const auto [q, a, b, w] = layer.conv.spec[t];
                const double coefficient = layer.conv.coefficients[t] * Y[long(edge) * 16 + b];
                for (int u = 0; u < C; ++u) out[q + u] += coefficient * x[a + u] * R[w + u];
            }
        }
    }
    A.resize(long(n) * 16 * C);
    // The export already folds division by avg_num_neighbors into this map.
    layer.output.forward(n, message.data(), message_width, A.data(), 16 * C, false);

    int basis_size = 16;
    for (const auto& term : layer.product.basis_spec) basis_size = std::max(basis_size, term[0] + 1);
    M.assign(long(n) * 4 * C, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        std::vector<double> B(long(basis_size) * C, 0.0);
        std::copy_n(A.data() + long(i) * 16 * C, 16 * C, B.data());
        for (int t = 0; t < layer.product.basis_spec.size(); ++t) {
            const auto [q, a, b] = layer.product.basis_spec[t];
            for (int u = 0; u < C; ++u)
                B[q * C + u] += layer.product.coefficients[t] * B[a * C + u] * B[b * C + u];
        }
        auto out = M.data() + long(i) * 4 * C;
        for (int t = 0; t < layer.product.output_spec.size(); ++t) {
            const auto [q, a, count] = layer.product.output_spec[t];
            const auto weights = layer.product.weights.data()
                                 + (long(node_types[i]) * layer.product.output_spec.size() + t) * C;
            for (int m = 0; m < count; ++m)
                for (int u = 0; u < C; ++u) out[(q + m) * C + u] += weights[u] * B[(a + m) * C + u];
        }
    }
    H.resize(long(n) * 4 * C);
    layer.product_linear.forward(n, M.data(), 4 * C, H.data(), 4 * C, false);
    // Batch each species through Polar's linear operator, which is vectorized
    // across atoms. The residual maps remain element-dependent in LSC.
    for (int type = 0; type < layer.skip.size(); ++type) {
        std::vector<int> rows;
        std::vector<double> gathered;
        for (int i = 0; i < n; ++i)
            if (node_types[i] == type) {
                rows.push_back(i);
                const auto first = input.data() + long(i) * input_width;
                gathered.insert(gathered.end(), first, first + input_width);
            }
        if (rows.empty()) continue;
        std::vector<double> residual(long(rows.size()) * 4 * C);
        layer.skip[type].forward(rows.size(), gathered.data(), input_width, residual.data(), 4 * C, false);
        for (int r = 0; r < rows.size(); ++r)
            for (int u = 0; u < 4 * C; ++u) H[long(rows[r]) * 4 * C + u] += residual[long(r) * 4 * C + u];
    }
}

void compute_source(Source& source, std::span<const int> node_types,
                    std::span<const int> senders, std::span<const int> receivers,
                    const std::vector<double>& formal, const std::vector<double>& H,
                    const std::vector<double>& radii, const std::vector<double>& Y,
                    const std::vector<int>& incoming_starts, const std::vector<int>& incoming_edges,
                    const std::vector<int>& outgoing_starts, const std::vector<int>& outgoing_edges,
                    std::vector<double>& features, std::vector<double>& density, std::vector<double>& flux)
{
    const int n = node_types.size(), count = senders.size();
    const auto species_oxidation = oxidation_embeddings(source.embedding, formal);
    std::vector<double> oxidation(long(n) * C);
    for (int i = 0; i < n; ++i)
        std::copy_n(species_oxidation.data() + node_types[i] * C, C, oxidation.data() + long(i) * C);
    features.resize(long(n) * 4 * C);
    source.modulation.forward(n, H.data(), oxidation.data(), features.data());
    source.sender_features.resize(long(n) * 4 * C);
    source.receiver_features.resize(long(n) * 4 * C);
    auto& sender_features = source.sender_features;
    auto& receiver_features = source.receiver_features;
    source.linear1.forward(n, features.data(), 4 * C, sender_features.data(), 4 * C, false);
    source.linear2.forward(n, features.data(), 4 * C, receiver_features.data(), 4 * C, false);

    std::vector<double> message(long(count) * C, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < count; ++e) {
        std::vector<double> R(source.radial.outputs);
        source.radial.projected_splines->evaluate(radii[e], R);
        const auto sender = sender_features.data() + long(senders[e]) * 4 * C;
        const auto receiver = receiver_features.data() + long(receivers[e]) * 4 * C;
        auto out = message.data() + long(e) * C;
        for (int t = 0; t < source.conv.spec.size(); ++t) {
            const auto [q, a, b, w] = source.conv.spec[t];
            const double coefficient = source.conv.coefficients[t] * Y[long(e) * 16 + b];
            for (int u = 0; u < C; ++u)
                out[q + u] += coefficient * (sender[a + u] + receiver[a + u]) * R[w + u];
        }
    }
    flux.resize(count);
    if (count > 0) source.readout.forward(count, message.data(), C, flux.data(), 1, false);
    for (auto& value : flux) value *= source.transfer_scale;
    density.resize(long(n) * 4);
    source.multipoles.forward(n, features.data(), 4 * C, density.data(), 4, false);
    // The local monopole map is unused by this family. Replace it with the
    // conservative divergence of the directed transfers, just as in PyTorch.
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double incoming = 0.0, outgoing = 0.0;
        for (int j = incoming_starts[i]; j < incoming_starts[i + 1]; ++j) incoming += flux[incoming_edges[j]];
        for (int j = outgoing_starts[i]; j < outgoing_starts[i + 1]; ++j) outgoing += flux[outgoing_edges[j]];
        density[long(i) * 4] = incoming - outgoing;
    }
}

void reverse_edge_geometry(int e, double radial_adj, const std::array<double, 16>& angular_adj,
                            const std::vector<double>& radii, const std::vector<double>& Y_grad,
                            std::span<const double> vectors, std::vector<double>& edge_adj)
{
    for (int d = 0; d < 3; ++d) {
        double gradient = radial_adj * vectors[3*e+d] / radii[e];
        for (int m = 0; m < 16; ++m) gradient += angular_adj[m] * Y_grad[(long(e)*3+d)*16+m];
        edge_adj[3*e+d] += gradient;
    }
}

void reverse_source(Source& source, std::span<const int> senders, std::span<const int> receivers,
                     const std::vector<int>& incoming_starts, const std::vector<int>& incoming_edges,
                     const std::vector<int>& outgoing_starts, const std::vector<int>& outgoing_edges,
                     const std::vector<double>& radii, const std::vector<double>& Y,
                     const std::vector<double>& Y_grad, std::span<const double> vectors,
                     const std::vector<double>& H, const std::vector<double>& density_adj,
                     const std::vector<double>& flux_adj, std::vector<double>& H_adj,
                     std::vector<double>& edge_adj)
{
    const int n = H.size() / (4*C), count = senders.size();
    std::vector<double> features_adj(long(n)*4*C, 0.0), multipoles_adj = density_adj;
    // Local monopoles were discarded by the forward source; only its dipoles
    // differentiate through this map. Monopoles seed the transfer divergence.
    for (int i = 0; i < n; ++i) multipoles_adj[4*i] = 0;
    source.multipoles.reverse(n, multipoles_adj.data(), 4, features_adj.data(), 4*C);
    std::vector<double> transfer_adj(count), message_adj(long(count)*C, 0.0);
    for (int e = 0; e < count; ++e)
        transfer_adj[e] = source.transfer_scale * (density_adj[4*receivers[e]] - density_adj[4*senders[e]] + flux_adj[e]);
    source.readout.reverse(count, transfer_adj.data(), 1, message_adj.data(), C);
    std::vector<double> sender_adj(long(n)*4*C, 0.0), receiver_adj(long(n)*4*C, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        std::vector<double> R(source.radial.outputs), dR(source.radial.outputs);
        for (int p = outgoing_starts[i]; p < outgoing_starts[i+1]; ++p) {
            const int e = outgoing_edges[p];
            source.radial.projected_splines->evaluate_derivs(radii[e], R, dR);
            const auto x = source.sender_features.data() + long(i)*4*C;
            const auto z = source.receiver_features.data() + long(receivers[e])*4*C;
            const auto message = message_adj.data() + long(e)*C;
            auto adj = sender_adj.data() + long(i)*4*C;
            double radial_adj = 0;
            std::array<double, 16> angular_adj = {};
            for (int t = 0; t < source.conv.spec.size(); ++t) {
                const auto [q, a, b, w] = source.conv.spec[t];
                const double coefficient = source.conv.coefficients[t];
                double dY = 0, dr = 0;
                for (int u = 0; u < C; ++u) {
                    const double g = message[q+u] * coefficient;
                    const double value = x[a+u] + z[a+u];
                    adj[a+u] += g * Y[long(e)*16+b] * R[w+u];
                    dY += g * value * R[w+u];
                    dr += g * value * dR[w+u];
                }
                angular_adj[b] += dY;
                radial_adj += Y[long(e)*16+b] * dr;
            }
            reverse_edge_geometry(e, radial_adj, angular_adj, radii, Y_grad, vectors, edge_adj);
        }
        // Gather receiver derivatives separately: both endpoint feature maps
        // contribute, but each worker owns its atom and writes without atomics.
        for (int p = incoming_starts[i]; p < incoming_starts[i+1]; ++p) {
            const int e = incoming_edges[p];
            source.radial.projected_splines->evaluate(radii[e], R);
            const auto message = message_adj.data() + long(e)*C;
            auto adj = receiver_adj.data() + long(i)*4*C;
            for (int t = 0; t < source.conv.spec.size(); ++t) {
                const auto [q, a, b, w] = source.conv.spec[t];
                const double factor = source.conv.coefficients[t] * Y[long(e)*16+b];
                for (int u = 0; u < C; ++u) adj[a+u] += message[q+u] * factor * R[w+u];
            }
        }
    }
    source.linear1.reverse(n, sender_adj.data(), 4*C, features_adj.data(), 4*C);
    source.linear2.reverse(n, receiver_adj.data(), 4*C, features_adj.data(), 4*C);
    // Formal oxidation states are fixed, hence their embedding adjoint has no
    // position dependence. Reuse Polar's modulation reverse for the H adjoint.
    std::vector<double> oxidation_adj(long(n)*C, 0.0);
    source.modulation.reverse(n, H.data(), features_adj.data(), H_adj.data(), oxidation_adj.data());
}

void reverse_layer(Layer& layer, std::span<const int> node_types,
                    std::span<const int> receivers, const std::vector<int>& outgoing_starts,
                    const std::vector<int>& outgoing_edges, const std::vector<double>& radii,
                    const std::vector<double>& Y, const std::vector<double>& Y_grad,
                    std::span<const double> vectors, const std::vector<double>& A,
                    const std::vector<double>& H_adj, std::vector<double>& input_adj,
                    std::vector<double>& edge_adj, int input_nodes=-1)
{
    const int n = node_types.size(), input_width = layer.up.off_in.back();
    const int up_width = layer.up.off_out.back(), message_width = layer.output.off_in.back();
    const int all = input_nodes<0?n:input_nodes;
    std::vector<double> M_adj(long(n)*4*C, 0.0), A_adj(long(n)*16*C, 0.0);
    layer.product_linear.reverse(n, H_adj.data(), 4*C, M_adj.data(), 4*C);
    for (int type = 0; type < layer.skip.size(); ++type) {
        std::vector<int> rows;
        std::vector<double> gathered;
        for (int i = 0; i < n; ++i)
            if (node_types[i] == type) {
                rows.push_back(i);
                const auto begin = H_adj.data() + long(i)*4*C;
                gathered.insert(gathered.end(), begin, begin+4*C);
            }
        if (rows.empty()) continue;
        std::vector<double> residual(long(rows.size())*input_width, 0.0);
        layer.skip[type].reverse(rows.size(), gathered.data(), 4*C, residual.data(), input_width);
        for (int r = 0; r < rows.size(); ++r)
            for (int u = 0; u < input_width; ++u) input_adj[long(rows[r])*input_width+u] += residual[long(r)*input_width+u];
    }
    int basis_size = 16;
    for (const auto& term : layer.product.basis_spec) basis_size = std::max(basis_size, term[0]+1);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // Reconstruct one atom's recursive basis from saved A. This follows
        // Polar's reverse recursion without retaining a large all-atom B cache.
        std::vector<double> B(long(basis_size)*C, 0.0), B_adj(B.size(), 0.0);
        std::copy_n(A.data()+long(i)*16*C, 16*C, B.data());
        for (int t = 0; t < layer.product.basis_spec.size(); ++t) {
            const auto [q, a, b] = layer.product.basis_spec[t];
            for (int u = 0; u < C; ++u) B[q*C+u] += layer.product.coefficients[t] * B[a*C+u] * B[b*C+u];
        }
        const auto g = M_adj.data()+long(i)*4*C;
        for (int t = 0; t < layer.product.output_spec.size(); ++t) {
            const auto [q, a, count] = layer.product.output_spec[t];
            const auto weights = layer.product.weights.data() + (long(node_types[i])*layer.product.output_spec.size()+t)*C;
            for (int m = 0; m < count; ++m)
                for (int u = 0; u < C; ++u) B_adj[(a+m)*C+u] += weights[u] * g[(q+m)*C+u];
        }
        for (int t = int(layer.product.basis_spec.size())-1; t >= 0; --t) {
            const auto [q, a, b] = layer.product.basis_spec[t];
            for (int u = 0; u < C; ++u) {
                const double g = layer.product.coefficients[t] * B_adj[q*C+u];
                B_adj[a*C+u] += g * B[b*C+u];
                B_adj[b*C+u] += g * B[a*C+u];
            }
        }
        std::copy_n(B_adj.data(), 16*C, A_adj.data()+long(i)*16*C);
    }
    std::vector<double> message_adj(long(n)*message_width, 0.0), up_adj(long(all)*up_width, 0.0);
    layer.output.reverse(n, A_adj.data(), 16*C, message_adj.data(), message_width);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < all; ++i) {
        std::vector<double> R(layer.radial.outputs), dR(layer.radial.outputs);
        const auto x = layer.up_features.data()+long(i)*up_width;
        auto adj = up_adj.data()+long(i)*up_width;
        for (int p = outgoing_starts[i]; p < outgoing_starts[i+1]; ++p) {
            const int e = outgoing_edges[p];
            const auto message = message_adj.data()+long(receivers[e])*message_width;
            layer.radial.projected_splines->evaluate_derivs(radii[e], R, dR);
            double radial_adj = 0;
            std::array<double, 16> angular_adj = {};
            for (int t = 0; t < layer.conv.spec.size(); ++t) {
                const auto [q, a, b, w] = layer.conv.spec[t];
                const double coefficient = layer.conv.coefficients[t];
                double dY = 0, dr = 0;
                for (int u = 0; u < C; ++u) {
                    const double g = message[q+u] * coefficient;
                    adj[a+u] += g * Y[long(e)*16+b] * R[w+u];
                    dY += g * x[a+u] * R[w+u];
                    dr += g * x[a+u] * dR[w+u];
                }
                angular_adj[b] += dY;
                radial_adj += Y[long(e)*16+b] * dr;
            }
            reverse_edge_geometry(e, radial_adj, angular_adj, radii, Y_grad, vectors, edge_adj);
        }
    }
    layer.up.reverse(all, up_adj.data(), up_width, input_adj.data(), input_width);
}

struct Electrostatics {
    double field_constant, sigma2, sigma, offset, epsilon;
    std::vector<double> a, reciprocal_self, realspace_self;
    Matrix displacements;
};

Electrostatics energy_parameters(const json& j)
{
    const std::string path = "electrostatics.energy_parameters";
    require(integer(field(j, "format_version", path), path + ".format_version") == 1,
            path + ".format_version", "requires version 1");
    Electrostatics result;
    result.field_constant = number(field(j, "field_constant", path), path + ".field_constant", true);
    const auto& reciprocal = field(j, "reciprocal", path);
    const auto rp = path + ".reciprocal";
    result.sigma2 = number(field(reciprocal, "sigma2", rp), rp + ".sigma2", true);
    result.a = numbers(field(reciprocal, "a", rp), rp + ".a", 2);
    require(result.a[0] > 0 and result.a[1] > 0, rp + ".a", "requires positive amplitudes");
    result.reciprocal_self = numbers(field(reciprocal, "self_overlap", rp), rp + ".self_overlap", 4);
    const auto& real = field(j, "realspace", path);
    const auto sp = path + ".realspace";
    result.sigma = number(field(real, "sigma", sp), sp + ".sigma", true);
    result.offset = number(field(real, "offset", sp), sp + ".offset", true);
    result.epsilon = number(field(real, "distance_epsilon", sp), sp + ".distance_epsilon", true);
    result.displacements = matrix(field(real, "offset_vectors", sp), sp + ".offset_vectors", 3, 3);
    result.realspace_self = numbers(field(real, "self_overlap", sp), sp + ".self_overlap", 4);
    for (int m = 0; m < 4; ++m)
        require(result.realspace_self[m] > 0 and result.reciprocal_self[m] > 0, path, "requires positive self overlaps");
    return result;
}

double self_energy(const std::vector<double>& density, const std::vector<double>& overlap)
{
    double result = 0;
    for (size_t i = 0; i < density.size(); ++i) result += 0.5 * density[i] * density[i] * overlap[i % 4];
    return result;
}

double realspace_energy(const Electrostatics& p, std::span<const double> xyz,
                        const std::vector<double>& density, bool include_self)
{
    const int n = xyz.size() / 3;
    // MACE-SCF represents each dipole by three displaced charges. Interactions
    // within one original atom are excluded, then Gaussian self terms are added.
    std::vector<std::array<double, 4>> charges(n);
    for (int i = 0; i < n; ++i) {
        auto& q = charges[i];
        q[1] = density[4 * i + 3] / p.offset;
        q[2] = density[4 * i + 1] / p.offset;
        q[3] = density[4 * i + 2] / p.offset;
        q[0] = density[4 * i] - q[1] - q[2] - q[3];
    }
    std::vector<double> rows(n, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    std::array<double, 3> delta;
                    for (int d = 0; d < 3; ++d)
                        delta[d] = xyz[3 * i + d] - xyz[3 * j + d]
                                   + (a == 0 ? 0 : p.displacements[a - 1][d])
                                   - (b == 0 ? 0 : p.displacements[b - 1][d]);
                    const double r = std::hypot(delta[0], delta[1], delta[2]);
                    rows[i] += charges[i][a] * charges[j][b] * std::erf(r * 0.5 / p.sigma) / (r + p.epsilon);
                }
    // i<j counts both directed half-weighted reference edges exactly once.
    double result = p.field_constant / (4 * std::numbers::pi) * std::accumulate(rows.begin(), rows.end(), 0.0);
    if (include_self) result += self_energy(density, p.realspace_self);
    return result;
}

struct ReciprocalCell {
    std::array<std::array<double, 3>, 3> basis;
    std::array<int, 3> bounds;
    double volume;
};

ReciprocalCell reciprocal_cell(const std::array<double, 9>& cell, double cutoff)
{
    ReciprocalCell result;
    auto cross = [&cell](int a, int b) {
        return std::array<double, 3>{cell[3*a+1]*cell[3*b+2]-cell[3*a+2]*cell[3*b+1],
                                    cell[3*a+2]*cell[3*b]-cell[3*a]*cell[3*b+2],
                                    cell[3*a]*cell[3*b+1]-cell[3*a+1]*cell[3*b]};
    };
    const auto c = cross(1, 2);
    const double det = cell[0]*c[0] + cell[1]*c[1] + cell[2]*c[2];
    require(finite_value(det) and std::abs(det) > 0, "cell", "periodic cell must be nonsingular");
    result.volume = std::abs(det);
    long long candidates = 1;
    for (int i = 0; i < 3; ++i) {
        auto b = cross((i+1)%3, (i+2)%3);
        for (int d = 0; d < 3; ++d) {
            b[d] *= 2 * std::numbers::pi / det;
            require(finite_value(b[d]), "cell", "invalid reciprocal cell");
        }
        result.basis[i] = b;
        // Match graph_longrange's half-space enumeration, including endpoints.
        const double bound = std::ceil(cutoff * std::hypot(cell[3*i], cell[3*i+1], cell[3*i+2]) / (2 * std::numbers::pi));
        require(finite_value(bound) and bound >= 1 and bound <= 1000000, "cell", "excessive reciprocal-space bounds");
        result.bounds[i] = static_cast<int>(bound);
        candidates *= 2LL * result.bounds[i];
        require(candidates <= std::numeric_limits<int>::max(), "cell", "excessive reciprocal-space grid");
    }
    return result;
}

std::vector<std::array<double, 3>> reciprocal_vectors(const ReciprocalCell& cell, double cutoff)
{
    std::vector<std::array<double, 3>> vectors;
    const auto& bounds = cell.bounds;
    for (int a = 0; a < bounds[0]; ++a)
        for (int b = a == 0 ? 0 : -bounds[1]; b < bounds[1]; ++b)
            for (int c = a == 0 and b == 0 ? 1 : -bounds[2]; c < bounds[2]; ++c) {
                std::array<double, 3> k;
                double norm2 = 0;
                for (int d = 0; d < 3; ++d) {
                    k[d] = a * cell.basis[0][d] + b * cell.basis[1][d] + c * cell.basis[2][d];
                    norm2 += k[d] * k[d];
                }
                if (norm2 <= cutoff * cutoff) vectors.push_back(k);
            }
    return vectors;
}

double reciprocal_energy(const Electrostatics& p, const ReciprocalCell& cell, double cutoff,
                         std::span<const double> xyz, const std::vector<double>& density, bool include_self)
{
    const auto vectors = reciprocal_vectors(cell, cutoff);
    std::vector<double> terms(vectors.size());
    #pragma omp parallel for
    for (long t = 0; t < vectors.size(); ++t) {
        const auto& k = vectors[t];
        const double norm2 = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
        const double gaussian = std::exp(-0.5 * p.sigma2 * norm2);
        const double monopole = p.a[0] * gaussian / std::sqrt(4 * std::numbers::pi);
        const double dipole = -p.a[1] * gaussian * std::sqrt(3 / (4 * std::numbers::pi));
        double re = 0, im = 0;
        for (size_t i = 0; i < xyz.size() / 3; ++i) {
            const double phase = xyz[3*i]*k[0] + xyz[3*i+1]*k[1] + xyz[3*i+2]*k[2];
            const double A = monopole * density[4*i];
            const double B = dipole * (density[4*i+3]*k[0] + density[4*i+1]*k[1] + density[4*i+2]*k[2]);
            const double cosine = std::cos(phase), sine = std::sin(phase);
            re += A * cosine + B * sine;
            im += B * cosine - A * sine;
        }
        terms[t] = (re*re + im*im) / norm2;
    }
    // Half reciprocal space removes the explicit 1/2, as in MACE-POLAR.
    double result = p.field_constant / cell.volume * std::accumulate(terms.begin(), terms.end(), 0.0);
    if (not include_self) result -= self_energy(density, p.reciprocal_self);
    return result;
}
// Analytic electrostatic adjoints. Density coefficients are held independent
// here; reverse_source below propagates their dependence on the local network.
void reverse_realspace(const Electrostatics& p, std::span<const double> xyz,
                       const std::vector<double>& density, bool include_self,
                       std::vector<double>& density_adj, std::vector<double>& position_adj,
                       std::array<double, 9>& strain)
{
    const int n = xyz.size() / 3;
    std::vector<std::array<double, 4>> charges(n);
    for (int i = 0; i < n; ++i) {
        auto& q = charges[i];
        q[1] = density[4*i+3] / p.offset;
        q[2] = density[4*i+1] / p.offset;
        q[3] = density[4*i+2] / p.offset;
        q[0] = density[4*i] - q[1] - q[2] - q[3];
    }
    const double prefactor = p.field_constant / (4 * std::numbers::pi);
    // Each worker owns an atom's density/position adjoints. All other original
    // atoms contribute; there is no 1/2 in a derivative of a pair energy.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        std::array<double, 4> charge_adj = {};
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    std::array<double, 3> delta;
                    for (int d = 0; d < 3; ++d)
                        delta[d] = xyz[3*i+d] - xyz[3*j+d]
                                   + (a == 0 ? 0 : p.displacements[a-1][d])
                                   - (b == 0 ? 0 : p.displacements[b-1][d]);
                    const double r = std::hypot(delta[0], delta[1], delta[2]);
                    const double smooth = std::erf(r * 0.5 / p.sigma);
                    const double denominator = r + p.epsilon;
                    charge_adj[a] += prefactor * smooth / denominator * charges[j][b];
                    if (r > 0) {
                        const double smooth_deriv = std::exp(-r*r / (4*p.sigma*p.sigma))
                                                    / (p.sigma * std::sqrt(std::numbers::pi));
                        const double radial = prefactor * charges[i][a] * charges[j][b]
                                              * (smooth_deriv / denominator - smooth / (denominator*denominator)) / r;
                        for (int d = 0; d < 3; ++d) position_adj[3*i+d] += radial * delta[d];
                    }
                }
        }
        density_adj[4*i] += charge_adj[0];
        density_adj[4*i+3] += (charge_adj[1] - charge_adj[0]) / p.offset;
        density_adj[4*i+1] += (charge_adj[2] - charge_adj[0]) / p.offset;
        density_adj[4*i+2] += (charge_adj[3] - charge_adj[0]) / p.offset;
        if (include_self)
            for (int m = 0; m < 4; ++m) density_adj[4*i+m] += density[4*i+m] * p.realspace_self[m];
    }
    // The Cartesian finite-displacement buffers stay fixed under strain, just
    // as in the reference; only original atom positions are deformed.
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) strain[3*a+b] += xyz[3*i+a] * position_adj[3*i+b];
}

void reverse_reciprocal(const Electrostatics& p, const ReciprocalCell& cell, double cutoff,
                        std::span<const double> xyz, const std::vector<double>& density, bool include_self,
                        std::vector<double>& density_adj, std::vector<double>& position_adj,
                        std::array<double, 9>& strain)
{
    struct Term {
        std::array<double, 3> k;
        double norm2, monopole, dipole, re, im;
        std::array<double, 9> strain;
    };
    const auto vectors = reciprocal_vectors(cell, cutoff);
    std::vector<Term> terms(vectors.size());
    const double prefactor = p.field_constant / cell.volume;
    const int n = xyz.size() / 3;
    #pragma omp parallel for
    for (long t = 0; t < vectors.size(); ++t) {
        auto& term = terms[t];
        const auto& k = vectors[t];
        term.k = k;
        term.norm2 = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
        const double gaussian = std::exp(-0.5 * p.sigma2 * term.norm2);
        term.monopole = p.a[0] * gaussian / std::sqrt(4 * std::numbers::pi);
        term.dipole = -p.a[1] * gaussian * std::sqrt(3 / (4 * std::numbers::pi));
        term.re = term.im = 0;
        std::array<double, 3> dipole_re = {}, dipole_im = {};
        constexpr int cartesian[3] = {3, 1, 2};
        for (int i = 0; i < n; ++i) {
            const double phase = xyz[3*i]*k[0] + xyz[3*i+1]*k[1] + xyz[3*i+2]*k[2];
            const double cosine = std::cos(phase), sine = std::sin(phase);
            const double A = term.monopole * density[4*i];
            const double B = term.dipole * (density[4*i+3]*k[0] + density[4*i+1]*k[1] + density[4*i+2]*k[2]);
            term.re += A*cosine + B*sine;
            term.im += B*cosine - A*sine;
            for (int d = 0; d < 3; ++d) {
                dipole_re[d] += term.dipole * density[4*i+cartesian[d]] * sine;
                dipole_im[d] += term.dipole * density[4*i+cartesian[d]] * cosine;
            }
        }
        const double magnitude = term.re*term.re + term.im*term.im;
        const double e = prefactor * magnitude / term.norm2;
        for (int a = 0; a < 3; ++a) {
            // Under a common deformation of positions and cell, k.r is
            // invariant. Only the basis, inverse k^2 and inverse volume vary.
            const double dre = dipole_re[a] - p.sigma2 * k[a] * term.re;
            const double dim = dipole_im[a] - p.sigma2 * k[a] * term.im;
            const double dk = 2 * prefactor / term.norm2 * (term.re*dre + term.im*dim)
                              - 2 * e * k[a] / term.norm2;
            for (int b = 0; b < 3; ++b) term.strain[3*a+b] = -dk*k[b] - (a == b ? e : 0.0);
        }
    }
    // Node-owned accumulation avoids atomics and atom-by-k tensor storage.
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (const auto& term : terms) {
            const auto& k = term.k;
            const double phase = xyz[3*i]*k[0] + xyz[3*i+1]*k[1] + xyz[3*i+2]*k[2];
            const double cosine = std::cos(phase), sine = std::sin(phase);
            const double weight = 2 * prefactor / term.norm2;
            const double dA = weight * (term.re*cosine - term.im*sine);
            const double dB = weight * (term.re*sine + term.im*cosine);
            density_adj[4*i] += dA * term.monopole;
            constexpr int cartesian[3] = {3, 1, 2};
            for (int d = 0; d < 3; ++d) density_adj[4*i+cartesian[d]] += dB * term.dipole * k[d];
            const double A = term.monopole * density[4*i];
            const double B = term.dipole * (density[4*i+3]*k[0] + density[4*i+1]*k[1] + density[4*i+2]*k[2]);
            const double re = A*cosine + B*sine, im = B*cosine - A*sine;
            const double dphase = weight * (term.re*im - term.im*re);
            for (int d = 0; d < 3; ++d) position_adj[3*i+d] += dphase * k[d];
        }
        if (not include_self)
            for (int m = 0; m < 4; ++m) density_adj[4*i+m] -= density[4*i+m] * p.reciprocal_self[m];
    }
    for (const auto& term : terms)
        for (int d = 0; d < 9; ++d) strain[d] += term.strain[d];
}
} // namespace

struct MACELSC::Parameters {
    std::optional<LSCGraph> graph;
    std::vector<int> serial_senders;
    LSCForwardState forward;
    LSCChargeExchange charge_exchange;
    LSCReverseExchange reverse_exchange;
    std::array<std::vector<double>,2> mpi_H_adj;
    std::vector<double> mpi_forces;
    std::array<std::vector<double>,2> partial_layer_density,all_source_features;
    std::vector<double> potential_halo;
    std::array<std::vector<double>,2> halo_features;
    std::vector<double> initial_all;
    int local_stage=0,reverse_stage=0;
    std::vector<double> positions,position_adj,edge_adj,flux_adj,Y_grad;
    std::array<std::vector<double>,2> H_adj;
    std::array<double,9> energy_cell{};
    std::array<bool,3> energy_pbc{};
    std::array<double,3> energy_field{};
    std::string energy_mode;
    LSCElectrostaticsHook electrostatics_hook;
    LSCElectrostaticsResult hook_result;
    std::unique_ptr<Electrostatics> electrostatics;
    Oxidation embedding;
    std::array<Layer, 2> layers;
    std::array<Source, 2> sources;
    std::vector<double> radii, Y;
    std::vector<int> starts, edges, outgoing_starts, outgoing_edges;
};

MACELSC::MACELSC(std::string filename) : parameters(std::make_unique<Parameters>())
{
    load_from_json(filename);
}

MACELSC::~MACELSC() = default;

std::array<std::vector<double>, 2> MACELSC::energy_readout_weights() const
{
    return {parameters->layers[0].readout.weights[0], parameters->layers[1].readout.weights[0]};
}


bool MACELSC::can_evaluate_energy() const
{
    return bool(parameters->electrostatics);
}

void MACELSC::set_electrostatics_hook(LSCElectrostaticsHook hook)
{
    clear_force_results();
    parameters->electrostatics_hook = std::move(hook);
    parameters->hook_result = {};
    parameters->reverse_stage=0;
    has_energy_results = has_force_results = has_stress_results = false;
    energy = electrostatic_energy = field_energy = 0;
    forces.clear(); edge_forces.clear(); position_forces.clear(); density_energy_gradient.clear();
    strain_derivative = {}; electrostatic_strain_derivative = {}; virial = {}; stress = {};
}

bool MACELSC::has_electrostatics_hook() const
{
    return bool(parameters->electrostatics_hook);
}

LSCElectrostaticsContext MACELSC::electrostatics_context(const std::array<double, 9>& cell,
    const std::array<bool, 3>& pbc, const std::string& requested_mode) const
{
    require(can_evaluate_energy(), "electrostatics.energy_parameters", "missing; re-export the checkpoint to enable energy evaluation");
    for (double value : cell) require(finite_value(value), "cell", "expected finite coordinates");
    std::string mode = requested_mode.empty() ? default_pbc_handling : requested_mode;
    const int periodic = std::count(pbc.begin(), pbc.end(), true);
    if (mode == "auto") mode = periodic == 0 ? "realspace" : "mixed_periodic";
    require(mode == "realspace" or mode == "pbc" or mode == "mixed_periodic", "pbc_handling", "supported modes: realspace, pbc, mixed_periodic, auto");
    require(mode == "realspace" or periodic == 3, "pbc", "reciprocal energy requires three periodic directions");
    LSCElectrostaticsContext result{cell, pbc, mode, {}};
    const auto& p = *parameters->electrostatics;
    auto& out = result.parameters;
    out.field_constant = p.field_constant; out.kspace_cutoff = kspace_cutoff; out.sigma2 = p.sigma2;
    std::copy_n(p.a.begin(), 2, out.a.begin());
    std::copy_n(p.reciprocal_self.begin(), 4, out.reciprocal_self.begin());
    std::copy_n(p.realspace_self.begin(), 4, out.realspace_self.begin());
    out.sigma = p.sigma; out.offset = p.offset; out.epsilon = p.epsilon;
    for (int i = 0; i < 3; ++i) std::copy_n(p.displacements[i].begin(), 3, out.displacements.begin() + 3*i);
    out.include_self = include_self_interaction;
    return result;
}

LSCElectrostaticsResult MACELSC::evaluate_direct_electrostatics(
    std::span<const double> positions, std::span<const double> density,
    const std::array<double, 9>& cell, const std::array<bool, 3>& pbc,
    const std::string& mode) const
{
    require(positions.size() % 3 == 0 and density.size() == positions.size()/3*4,
            "electrostatics", "expected positions[N,3] and density[N,4]");
    for (double x : positions) require(finite_value(x), "positions", "expected finite coordinates");
    for (double x : density) require(finite_value(x), "density", "expected finite coefficients");
    const auto context = electrostatics_context(cell, pbc, mode);
    const auto& p = *parameters->electrostatics;
    const std::vector<double> rho(density.begin(), density.end());
    LSCElectrostaticsResult result;
    result.density_gradient.assign(density.size(), 0);
    result.position_gradient.assign(positions.size(), 0);
    if (context.mode == "realspace") {
        result.energy = realspace_energy(p, positions, rho, include_self_interaction);
        reverse_realspace(p, positions, rho, include_self_interaction, result.density_gradient,
                         result.position_gradient, result.strain_derivative);
    } else {
        const auto reciprocal = reciprocal_cell(cell, kspace_cutoff);
        result.energy = reciprocal_energy(p, reciprocal, kspace_cutoff, positions, rho, include_self_interaction);
        reverse_reciprocal(p, reciprocal, kspace_cutoff, positions, rho, include_self_interaction,
                          result.density_gradient, result.position_gradient, result.strain_derivative);
    }
    return result;
}

void MACELSC::compute_energy(std::span<const int> node_types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> edge_vectors,std::span<const double> positions,
    const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode,const std::array<double,3>& field)
{
    require(can_evaluate_energy(),"electrostatics.energy_parameters","missing; re-export the checkpoint to enable energy evaluation");
    compute_charges(node_types,senders,receivers,edge_vectors,positions);evaluate_energy(cell,pbc,mode,field);
}
double MACELSC::evaluate_energy(const std::array<double,9>& cell,const std::array<bool,3>& pbc,
    const std::string& pbc_handling,const std::array<double,3>& external_field)
{
    clear_force_results();has_energy_results=false;energy=electrostatic_energy=field_energy=0;
    require(has_charge_results,"stages","evaluate charge sources before energy");
    require(!parameters->charge_exchange.active(),"MPI","distributed electrostatic energy and reverse communication are not implemented yet");
    const auto node_types=graph_layout().owned_types();const auto& positions=parameters->positions;
    require(can_evaluate_energy(), "electrostatics.energy_parameters", "missing; re-export the checkpoint to enable energy evaluation");
    for (double value : cell) require(finite_value(value), "cell", "expected finite coordinates");
    for (double value : external_field) require(finite_value(value), "external_field", "expected finite components");
    const int periodic = std::count(pbc.begin(), pbc.end(), true);
    std::string mode = pbc_handling.empty() ? default_pbc_handling : pbc_handling;
    if (mode == "auto") mode = periodic == 0 ? "realspace" : "mixed_periodic";
    const bool real = mode == "realspace";
    require(real or mode == "pbc" or mode == "mixed_periodic", "pbc_handling", "supported modes: realspace, pbc, mixed_periodic, auto");
    require(real or periodic == 3, "pbc", "reciprocal energy currently requires three periodic directions; slab and molecule-in-box corrections are not implemented");
    ReciprocalCell reciprocal;
    if (not real) reciprocal = reciprocal_cell(cell, kspace_cutoff);

    const auto& p = *parameters->electrostatics;
    if (parameters->electrostatics_hook) {
        // Compute/cache all electrostatic derivatives once, as Polar does in
        // its Coulomb hook. Energy-only calls can be followed by a reverse pass.
        parameters->hook_result = parameters->electrostatics_hook(
            {positions, density_coefficients, electrostatics_context(cell, pbc, mode)});
        validate_lsc_electrostatics_result(parameters->hook_result, node_types.size());
        electrostatic_energy = parameters->hook_result.energy;
    } else {
    electrostatic_energy = real ? realspace_energy(p, positions, density_coefficients, include_self_interaction)
                               : reciprocal_energy(p, reciprocal, kspace_cutoff, positions, density_coefficients, include_self_interaction);
    }
    field_energy = std::inner_product(dipole.begin(), dipole.end(), external_field.begin(), 0.0);
    energy = local_energy + electrostatic_energy + field_energy;
    parameters->energy_cell=cell;parameters->energy_pbc=pbc;parameters->energy_field=external_field;parameters->energy_mode=mode;
    has_energy_results = true;    return energy;
}

void MACELSC::compute_energy_forces(std::span<const int> types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> vectors,std::span<const double> positions,
    const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode,const std::array<double,3>& field,bool stress)
{
    compute_energy(types,senders,receivers,vectors,positions,cell,pbc,mode,field);evaluate_forces(stress);
}
void MACELSC::evaluate_forces(bool stress)
{
    reverse_charges();reverse_second_layer();reverse_first_layer();finalize_forces(stress);
}
void MACELSC::reverse_charges()
{
    require(!parameters->charge_exchange.active(),"stages","use distributed reverse after staged MPI sources");
    clear_force_results();require(has_energy_results,"stages","evaluate energy before reverse sources");
    const auto node_types=graph_layout().owned_types();const auto& senders=parameters->serial_senders;
    const auto& receivers=graph_layout().receivers();const auto& edge_vectors=graph_layout().edge_vectors();
    const auto& positions=parameters->positions;const auto& cell=parameters->energy_cell;
    const auto& external_field=parameters->energy_field;const int n=node_types.size(),count=senders.size();
    density_energy_gradient.assign(long(n)*4, 0.0);
    auto& position_adj=parameters->position_adj;auto& edge_adj=parameters->edge_adj;
    position_adj.assign(long(n)*3,0.0);edge_adj.assign(long(count)*3,0.0);
    const auto& mode=parameters->energy_mode;
    const auto& p = *parameters->electrostatics;
    if (parameters->electrostatics_hook) {
        density_energy_gradient = parameters->hook_result.density_gradient;
        position_adj = parameters->hook_result.position_gradient;
        electrostatic_strain_derivative = parameters->hook_result.strain_derivative;
    } else if (mode == "realspace")
        reverse_realspace(p, positions, density_coefficients, include_self_interaction,
                          density_energy_gradient, position_adj, electrostatic_strain_derivative);
    else
        reverse_reciprocal(p, reciprocal_cell(cell, kspace_cutoff), kspace_cutoff, positions,
                           density_coefficients, include_self_interaction, density_energy_gradient,
                           position_adj, electrostatic_strain_derivative);
    strain_derivative = electrostatic_strain_derivative;
    // E_field = F . (sum FQ*r + sum flux*image_vector + sum atomic dipoles).
    // Preserve the periodic polarization branch; predicted q*r is not equivalent.
    constexpr int cartesian[3] = {3, 1, 2};
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) {
            density_energy_gradient[4*i+cartesian[d]] += external_field[d];
            const double g = formal_charges[node_types[i]] * external_field[d];
            position_adj[3*i+d] += g;
            for (int a = 0; a < 3; ++a) strain_derivative[3*a+d] += positions[3*i+a] * g;
        }
    auto& flux_adj=parameters->flux_adj;flux_adj.assign(count,0.0);
    for (int e = 0; e < count; ++e)
        for (int d = 0; d < 3; ++d) {
            flux_adj[e] += external_field[d] * edge_vectors[3*e+d];
            edge_adj[3*e+d] += edge_fluxes[e] * external_field[d];
        }
    for(auto& gradient:parameters->H_adj)gradient.clear();
    if (n > 0) {
        auto& Y_grad=parameters->Y_grad;Y_grad.resize(long(count)*3*16);
        if (count > 0) {
            sphericart::SphericalHarmonics<double> harmonics(3);
            harmonics.compute_array_with_gradients(edge_vectors.data(), edge_vectors.size(),
                parameters->Y.data(), parameters->Y.size(), Y_grad.data(), Y_grad.size());
        }
        auto& H_adj=parameters->H_adj;
        const std::vector<double> readout_adj(n, 1.0);
        for (int t = 0; t < 2; ++t) {
            H_adj[t].assign(long(n)*4*C, 0.0);
            parameters->layers[t].readout.reverse(n, readout_adj.data(), 1, H_adj[t].data(), 4*C);
            reverse_source(parameters->sources[t], senders, receivers, parameters->starts, parameters->edges,
                parameters->outgoing_starts, parameters->outgoing_edges, parameters->radii, parameters->Y,
                Y_grad, edge_vectors, t == 0 ? H1 : H2, density_energy_gradient, flux_adj, H_adj[t], edge_adj);
        }
    }
    parameters->reverse_stage=1;
}
void MACELSC::reverse_second_layer()
{
    require(parameters->reverse_stage==1,"stages","reverse sources before second layer");
    const auto node_types=graph_layout().owned_types();const auto& receivers=graph_layout().receivers();
    const auto& edge_vectors=graph_layout().edge_vectors();const int n=node_types.size();
    auto& H_adj=parameters->H_adj;auto& edge_adj=parameters->edge_adj;auto& Y_grad=parameters->Y_grad;
    if(n){
        reverse_layer(parameters->layers[1], node_types, receivers, parameters->outgoing_starts,
            parameters->outgoing_edges, parameters->radii, parameters->Y, Y_grad, edge_vectors,
            A1, H_adj[1], H_adj[0], edge_adj);
    }
    parameters->reverse_stage=2;
}
void MACELSC::reverse_first_layer()
{
    require(parameters->reverse_stage==2,"stages","reverse second layer before first layer");
    const auto node_types=graph_layout().owned_types();const auto& receivers=graph_layout().receivers();
    const auto& edge_vectors=graph_layout().edge_vectors();const int n=node_types.size();
    auto& H_adj=parameters->H_adj;auto& edge_adj=parameters->edge_adj;auto& Y_grad=parameters->Y_grad;
    if(n){
        // Initial species/formal-charge embeddings are independent of geometry.
        std::vector<double> initial_adj(long(n)*C, 0.0);
        reverse_layer(parameters->layers[0], node_types, receivers, parameters->outgoing_starts,
            parameters->outgoing_edges, parameters->radii, parameters->Y, Y_grad, edge_vectors,
            A0, H_adj[0], initial_adj, edge_adj);
    }
    parameters->reverse_stage=3;
}
void MACELSC::finalize_forces(bool compute_stress)
{
    require(parameters->reverse_stage==3,"stages","reverse first layer before force assembly");
    const auto node_types=graph_layout().owned_types();const auto& senders=parameters->serial_senders;
    const auto& receivers=graph_layout().receivers();const auto& edge_vectors=graph_layout().edge_vectors();
    const auto& positions=parameters->positions;const auto& cell=parameters->energy_cell;
    const auto& external_field=parameters->energy_field;const int n=node_types.size(),count=senders.size();
    const auto& position_adj=parameters->position_adj;const auto& edge_adj=parameters->edge_adj;
    double volume=0;
    if(compute_stress){volume=std::abs(cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])-cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])+cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]));
        require(finite_value(volume) && volume>0,"cell","stress requires a nonsingular cell");}
    forces.resize(long(n)*3); position_forces.resize(long(n)*3); edge_forces.resize(long(count)*3);
    for (long k = 0; k < long(n)*3; ++k) forces[k] = position_forces[k] = -position_adj[k];
    for (int e = 0; e < count; ++e)
        for (int d = 0; d < 3; ++d) {
            const double f = -edge_adj[3*e+d];
            edge_forces[3*e+d] = f;
            forces[3*receivers[e]+d] += f;
            forces[3*senders[e]+d] -= f;
            for (int a = 0; a < 3; ++a) strain_derivative[3*a+d] += edge_vectors[3*e+a] * edge_adj[3*e+d];
        }
    // MACE differentiates a symmetric displacement. ASE stress is +dE/dstrain/V;
    // Polar/LAMMPS virial has the opposite sign and a different Voigt ordering.
    const auto unsymmetrized = strain_derivative;
    const auto electrostatic_unsymmetrized = electrostatic_strain_derivative;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            strain_derivative[3*a+b] = 0.5 * (unsymmetrized[3*a+b] + unsymmetrized[3*b+a]);
            electrostatic_strain_derivative[3*a+b] = 0.5 * (electrostatic_unsymmetrized[3*a+b] + electrostatic_unsymmetrized[3*b+a]);
        }
    constexpr int polar_order[6] = {0, 4, 8, 1, 2, 5};
    constexpr int ase_order[6] = {0, 4, 8, 5, 2, 1};
    for (int k = 0; k < 6; ++k) {
        virial[k] = -strain_derivative[polar_order[k]];
        if (compute_stress) stress[k] = strain_derivative[ase_order[k]] / volume;
    }
    for (double value : forces) require(finite_value(value), "forces", "nonfinite prediction");
    for (double value : strain_derivative) require(finite_value(value), "strain_derivative", "nonfinite prediction");
    for (double value : stress) require(finite_value(value), "stress", "nonfinite prediction");
    has_force_results = true;
    has_stress_results = compute_stress;
    parameters->reverse_stage=4;
}
std::vector<double> MACELSC::node_energies() const
{
    require(has_force_results,"energies","evaluate forces before partitioning total energy");
    const auto& graph=graph_layout();
    require(!graph.uses_remote_ghosts(),"energies","use the distributed pair for remote ghosts");
    const auto types=graph.owned_types();
    const auto& senders=parameters->serial_senders;const auto& receivers=graph.receivers();
    const auto& vectors=graph.edge_vectors();const auto& positions=parameters->positions;
    const auto& field=parameters->energy_field;
    auto out=local_node_energies;
    constexpr int cartesian[3]={3,1,2};
    for(size_t i=0;i<types.size();++i){
        for(int m=0;m<4;++m){
            double gradient=density_energy_gradient[4*i+m];
            if(m>0)gradient-=field[m==1?1:(m==2?2:0)];
            out[i]+=.5*density_coefficients[4*i+m]*gradient;
        }
        for(int d=0;d<3;++d)
            out[i]+=field[d]*(formal_charges[types[i]]*positions[3*i+d]
                             +density_coefficients[4*i+cartesian[d]]);
    }
    for(size_t e=0;e<senders.size();++e){
        double value=0;
        for(int d=0;d<3;++d)value+=.5*edge_fluxes[e]*field[d]*vectors[3*e+d];
        out[senders[e]]+=value;out[receivers[e]]+=value;
    }
    return out;
}
std::span<double> MACELSC::layer_features(int layer)
{
    require(layer==1 || layer==2,"layer","expected layer 1 or 2");
    require(parameters->local_stage>=layer,"stages","requested layer features are not ready");
    return layer==1?std::span<double>(H1):std::span<double>(H2);
}
std::span<double> MACELSC::layer_feature_gradients(int layer)
{
    require(layer==1 || layer==2,"layer","expected layer 1 or 2");
    require(parameters->reverse_stage>=1,"stages","reverse sources before accessing feature gradients");
    return parameters->H_adj[layer-1];
}
void MACELSC::clear_force_results()
{
    parameters->reverse_exchange.reset();parameters->mpi_forces.clear();
    parameters->reverse_stage=0;has_force_results=has_stress_results=false;
    forces.clear();edge_forces.clear();position_forces.clear();density_energy_gradient.clear();
    strain_derivative={};electrostatic_strain_derivative={};virial={};stress={};
}

void MACELSC::clear_charge_results()
{
    parameters->reverse_exchange.reset();parameters->mpi_forces.clear();
    parameters->charge_exchange.reset();parameters->potential_halo.clear();
    parameters->reverse_stage=0;
    for (auto& values : source_features) values.clear();
    for (auto& values : layer_density_coefficients) values.clear();
    for (auto& values : layer_edge_fluxes) values.clear();
    density_coefficients.clear(); partial_charges.clear(); partial_dipoles.clear(); edge_fluxes.clear();
    dipole = {}; formal_charge_dipole = {}; transfer_dipole = {}; atomic_dipole_sum = {};
    has_charge_results = false;
    has_energy_results = false;
    has_force_results = has_stress_results = false;
    forces.clear(); edge_forces.clear(); position_forces.clear(); density_energy_gradient.clear();
    strain_derivative = {}; electrostatic_strain_derivative = {}; virial = {}; stress = {};
    energy = electrostatic_energy = field_energy = 0.0;
}

void MACELSC::prepare_graph(const LSCGraph& input)
{
    const LSCGraph graph=input;
    clear_charge_results();has_local_results=false;local_energy=0;
    parameters->graph.reset();parameters->local_stage=0;parameters->forward=LSCForwardState{};
    for(int type:graph.types())require(type<num_elements,"node_types","species index is out of range");
    auto resolved=graph.feature_senders();
    parameters->forward.reset(graph);
    for(auto& h:parameters->halo_features)h.clear();parameters->initial_all.clear();
    const auto node_types=graph.owned_types();const auto& senders=resolved;
    const auto& receivers=graph.receivers();const auto& edge_vectors=graph.edge_vectors();
    require(node_types.size() <= std::numeric_limits<int>::max(), "node_types", "too many nodes");
    require(senders.size() == receivers.size() and senders.size() <= std::numeric_limits<int>::max(),
            "edge_index", "inconsistent or excessive edge counts");
    require(edge_vectors.size() == 3 * senders.size(), "edge_vectors", "expected one xyz vector per edge");
    const int n = node_types.size(), count = senders.size();
    for (int type : node_types) require(type >= 0 and type < num_elements, "node_types", "species index is out of range");
    std::vector<double> radii(count);
    std::vector<int> starts(size_t(n) + 1, 0);
    for (int e = 0; e < count; ++e) {
        require(senders[e] >= 0 and senders[e] < graph.num_nodes() and receivers[e] >= 0 and receivers[e] < n,
                "edge_index", "atom index is out of range");
        for (int d = 0; d < 3; ++d)
            require(finite_value(edge_vectors[long(e) * 3 + d]), "edge_vectors", "expected finite coordinates");
        radii[e] = std::hypot(edge_vectors[long(e) * 3], edge_vectors[long(e) * 3 + 1], edge_vectors[long(e) * 3 + 2]);
        require(finite_value(radii[e]) and radii[e] >= radial_spline_r_min and radii[e] <= r_cut,
                "edge_vectors", "edge distance is outside [r_min, r_cut]");
        ++starts[receivers[e] + 1];
    }
    std::partial_sum(starts.begin(), starts.end(), starts.begin());
    std::vector<int> edges(count), cursor = starts;
    for (int e = 0; e < count; ++e) edges[cursor[receivers[e]]++] = e;
    parameters->radii.clear(); parameters->Y.clear(); parameters->starts.clear(); parameters->edges.clear();
    parameters->outgoing_starts.clear(); parameters->outgoing_edges.clear();
    initial_features.clear(); A0.clear(); A1.clear(); M0.clear(); M1.clear(); H1.clear(); H2.clear();
    local_node_energies.clear();
    for (auto& energies : layer_node_energies) energies.clear();
    local_energy = 0.0;
    if (n == 0) { parameters->graph=graph;parameters->serial_senders=std::move(resolved);parameters->radii=std::move(radii);parameters->starts=std::move(starts);parameters->edges=std::move(edges);return; }

    // Initial oxidation embedding is constant for each species in this family.
    const auto oxidation_features = oxidation_embeddings(parameters->embedding, formal_charges);
    initial_features.resize(long(n) * C);
    for (int i = 0; i < n; ++i)
        for (int u = 0; u < C; ++u)
            initial_features[long(i) * C + u] = node_embedding[node_types[i]][u] + oxidation_features[node_types[i] * C + u];

    if(graph.uses_remote_ghosts()){
        parameters->initial_all.resize(long(graph.num_nodes())*C);
        for(int i=0;i<graph.num_nodes();++i)for(int u=0;u<C;++u)
            parameters->initial_all[long(i)*C+u]=node_embedding[graph.types()[i]][u]+oxidation_features[graph.types()[i]*C+u];
    }
    std::vector<double> Y(long(count) * 16);
    if (count > 0) {
        sphericart::SphericalHarmonics<double> harmonics(3);
        harmonics.compute_array(edge_vectors.data(), edge_vectors.size(), Y.data(), Y.size());
        // sphericart's integral harmonics with xyz match e3nn's harmonics with
        // the model's [y,z,x] coordinate permutation. The exported radial scale
        // 2*sqrt(pi) supplies e3nn's component normalization exactly once.
        for (auto& layer : parameters->layers) prepare_radial(layer.radial, radial_spline_r_min, r_cut);
    }
    parameters->graph=graph;parameters->serial_senders=std::move(resolved);
    parameters->radii=std::move(radii);parameters->Y=std::move(Y);
    parameters->starts=std::move(starts);parameters->edges=std::move(edges);
}

const LSCGraph& MACELSC::graph_layout()const
{
    if(!parameters->graph)throw std::runtime_error("MACELSC: prepare a graph first");return *parameters->graph;
}
std::string MACELSC::evaluation_stage()const
{
    if(has_force_results)return "forces";
    if(parameters->reverse_stage)return parameters->reverse_stage==1?"reverse_sources":parameters->reverse_stage==2?"reverse_second_layer":"reverse_first_layer";
    if(has_energy_results)return "energy";if(has_charge_results)return "sources";if(has_local_results)return "readouts";
    if(parameters->local_stage)return parameters->local_stage==1?"first_layer":"second_layer";
    return parameters->graph?"graph":"empty";
}
void MACELSC::compute_first_layer()
{
    const auto& graph=graph_layout();clear_charge_results();has_local_results=false;local_energy=0;
    parameters->forward.reset(graph);
    for(auto& h:parameters->halo_features)h.clear();
    parameters->local_stage=0;H1.clear();H2.clear();A0.clear();A1.clear();M0.clear();M1.clear();
    local_node_energies.clear();for(auto& e:layer_node_energies)e.clear();
    if(graph.num_owned())compute_layer(parameters->layers[0],graph.owned_types(),parameters->serial_senders,
        parameters->starts,parameters->edges,parameters->radii,parameters->Y,
        graph.uses_remote_ghosts()?parameters->initial_all:initial_features,A0,M0,H1,
        graph.uses_remote_ghosts()?graph.num_nodes():graph.num_owned());
    parameters->local_stage=1;publish_forward_features(1);
}
void MACELSC::compute_second_layer()
{
    require(parameters->local_stage==1,"stages","compute first layer before second layer");
    const auto& graph=graph_layout();complete_forward_exchange(1);
    if(graph.num_owned())compute_layer(parameters->layers[1],graph.owned_types(),parameters->serial_senders,
        parameters->starts,parameters->edges,parameters->radii,parameters->Y,
        graph.uses_remote_ghosts()?parameters->halo_features[0]:H1,A1,M1,H2,
        graph.uses_remote_ghosts()?graph.num_nodes():graph.num_owned());
    parameters->local_stage=2;publish_forward_features(2);
}
double MACELSC::compute_readouts()
{
    require(parameters->local_stage==2,"stages","compute second layer before readouts");
    complete_forward_exchange(2);
    const auto node_types=graph_layout().owned_types();const int n=node_types.size();
    if(n){
    layer_node_energies[0].resize(n); layer_node_energies[1].resize(n);
    parameters->layers[0].readout.forward(n, H1.data(), 4 * C, layer_node_energies[0].data(), 1, false);
    parameters->layers[1].readout.forward(n, H2.data(), 4 * C, layer_node_energies[1].data(), 1, false);
    local_node_energies.resize(n);
    for (int i = 0; i < n; ++i)
        local_node_energies[i] = atomic_energies[node_types[i]] + layer_node_energies[0][i] + layer_node_energies[1][i];
    local_energy = std::accumulate(local_node_energies.begin(), local_node_energies.end(), 0.0);
    require(finite_value(local_energy), "local_energy", "nonfinite local prediction");
    }
    parameters->local_stage=3;has_local_results=true;return local_energy;
}
void MACELSC::compute_local(std::span<const int> types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> vectors)
{
    try {
        prepare_graph(LSCGraph(types,senders,receivers,vectors));
    } catch (const std::invalid_argument& error) {
        // Preserve the established one-call API's diagnostic prefix; the new
        // graph API retains its own more specific ownership/indexing messages.
        const std::string message=error.what();
        if(message.starts_with("MACELSC:"))throw;
        throw std::invalid_argument("MACELSC: "+message);
    }
    compute_first_layer();compute_second_layer();compute_readouts();
}

void MACELSC::compute_charges(std::span<const int> node_types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> edge_vectors,std::span<const double> positions)
{
    require(positions.size()==3*node_types.size(),"positions","expected one xyz position per atom");
    for(double value:positions)require(finite_value(value),"positions","expected finite coordinates");
    compute_local(node_types,senders,receivers,edge_vectors);evaluate_charges(positions);
}
std::array<double,3> MACELSC::evaluate_charges(std::span<const double> positions)
{
    clear_charge_results();require(has_local_results,"stages","compute readouts before sources");
    require(!graph_layout().uses_remote_ghosts(),"MPI","remote charge accumulation requires the staged MPI source API; reverse communication is not implemented yet");
    require(positions.size()==3*size_t(graph_layout().num_owned()),"positions","expected one xyz position per owned atom");
    for(double value:positions)require(finite_value(value),"positions","expected finite coordinates");
    const auto node_types=graph_layout().owned_types();const auto& senders=parameters->serial_senders;
    const auto& receivers=graph_layout().receivers();const auto& edge_vectors=graph_layout().edge_vectors();
    const int n = node_types.size(), count = senders.size();
    parameters->positions.assign(positions.begin(),positions.end());
    if (n == 0) { has_charge_results = true; return dipole; }
    std::vector<int> outgoing_starts(size_t(n) + 1, 0);
    for (int sender : senders) ++outgoing_starts[sender + 1];
    std::partial_sum(outgoing_starts.begin(), outgoing_starts.end(), outgoing_starts.begin());
    std::vector<int> outgoing_edges(count), cursor = outgoing_starts;
    for (int e = 0; e < count; ++e) outgoing_edges[cursor[senders[e]]++] = e;
    density_coefficients.assign(long(n) * 4, 0.0);
    for (int i = 0; i < n; ++i) density_coefficients[long(i) * 4] = formal_charges[node_types[i]];
    edge_fluxes.assign(count, 0.0);
    for (int t = 0; t < 2; ++t) {
        auto& source = parameters->sources[t];
        if (count > 0) prepare_radial(source.radial, radial_spline_r_min, r_cut);
        compute_source(source, node_types, senders, receivers, formal_charges, t == 0 ? H1 : H2,
                       parameters->radii, parameters->Y, parameters->starts, parameters->edges,
                       outgoing_starts, outgoing_edges, source_features[t], layer_density_coefficients[t], layer_edge_fluxes[t]);
        for (long k = 0; k < long(n) * 4; ++k) density_coefficients[k] += layer_density_coefficients[t][k];
        for (int e = 0; e < count; ++e) edge_fluxes[e] += layer_edge_fluxes[t][e];
    }
    partial_charges.resize(n); partial_dipoles.resize(long(n) * 3);
    // Internal density order is [q,dy,dz,dx]; public atomic/total dipoles use xyz.
    constexpr int cartesian[3] = {3, 1, 2};
    for (int i = 0; i < n; ++i) {
        partial_charges[i] = density_coefficients[long(i) * 4];
        for (int d = 0; d < 3; ++d) {
            const double value = density_coefficients[long(i) * 4 + cartesian[d]];
            partial_dipoles[long(i) * 3 + d] = value;
            atomic_dipole_sum[d] += value;
            formal_charge_dipole[d] += formal_charges[node_types[i]] * positions[long(i) * 3 + d];
        }
    }
    // All directed edges contribute once, without a 1/2 factor. Image vectors
    // retain LSC's periodic polarization branch, unlike sum_i q_i * wrapped_r_i.
    for (int e = 0; e < count; ++e)
        for (int d = 0; d < 3; ++d) transfer_dipole[d] += edge_fluxes[e] * edge_vectors[long(e) * 3 + d];
    for (int d = 0; d < 3; ++d) {
        dipole[d] = formal_charge_dipole[d] + transfer_dipole[d] + atomic_dipole_sum[d];
        require(finite_value(dipole[d]), "dipole", "nonfinite prediction");
    }
    for (double value : density_coefficients) require(finite_value(value), "density_coefficients", "nonfinite prediction");
    parameters->outgoing_starts = std::move(outgoing_starts);
    parameters->outgoing_edges = std::move(outgoing_edges);
    has_charge_results = true;    return dipole;
}


namespace {
LSCModelData read_lsc_model(const std::string& filename)
{
    std::ifstream stream(filename);
    if (not stream) throw std::runtime_error("MACELSC: cannot open " + filename);
    LSCModelData model;
    try {
        model.document = json::parse(stream);
    } catch (const json::exception& error) {
        fail("JSON", "cannot parse " + filename + ": " + error.what());
    }
    return model;
}
} // namespace

MACELSC::MACELSC(const LSCModelData& model) : parameters(std::make_unique<Parameters>())
{
    load_parameters(model);
}

LSCModelData read_validated_lsc_model(const std::string& filename)
{
    auto model = read_lsc_model(filename);
    const MACELSC validated(model);
    return model;
}

void MACELSC::load_from_json(const std::string& filename)
{
    load_parameters(read_lsc_model(filename));
}

void MACELSC::load_parameters(const LSCModelData& model)
{
    const auto& file = model.document;
    format_version = integer(field(file, "format_version"), "format_version");
    require(format_version == 3, "format_version", "expected version 3; extract the model again");
    model_type = text(field(file, "model_type"), "model_type");
    require(model_type == "LocalSplitCharges", "model_type", "expected LocalSplitCharges");
    export_status = text(field(file, "export_status"), "export_status");
    require(export_status == "parameters_and_splines", "export_status", "requires parameters_and_splines; architecture-only files cannot be loaded");
    num_channels = integer(field(file, "num_channels"), "num_channels");
    l_max = integer(field(file, "l_max"), "l_max");
    L_max = integer(field(file, "L_max"), "L_max");
    require(num_channels == C and l_max == 3 and L_max == 1, "architecture", "requires 128 channels, l_max=3 and L_max=1");
    feature_layout = text(field(file, "feature_layout"), "feature_layout");
    require(feature_layout == "[l][m][channel]", "feature_layout", "unsupported feature order");
    r_cut = number(field(file, "r_cut"), "r_cut", true);
    radial_spline_r_min = number(field(file, "radial_spline_r_min"), "radial_spline_r_min", true);
    require(radial_spline_r_min < r_cut, "radial_spline_r_min", "must be smaller than r_cut");
    radial_spline_scale = number(field(file, "radial_spline_scale"), "radial_spline_scale", true);
    require(std::abs(radial_spline_scale - 2 * std::sqrt(std::numbers::pi)) < 1e-12, "radial_spline_scale", "expected 2*sqrt(pi)");
    require(not boolean(field(file, "radial_pair_dependent"), "radial_pair_dependent"), "radial_pair_dependent", "requires shared radial tables");
    const auto& options = field(file, "export_options");
    const int knots = integer(field(options, "num_spline_points", "export_options"), "export_options.num_spline_points", 2);
    require(number(field(options, "r_min", "export_options"), "export_options.r_min") == radial_spline_r_min,
            "export_options.r_min", "does not match radial_spline_r_min");
    num_elements = integer(field(file, "num_elements"), "num_elements", 1, 118);
    source_num_elements = integer(field(file, "source_num_elements"), "source_num_elements", num_elements, 118);
    atomic_numbers = integers(field(file, "atomic_numbers"), "atomic_numbers", num_elements);
    source_element_indices = integers(field(file, "source_element_indices"), "source_element_indices", num_elements);
    std::set<int> indices;
    for (int i = 0; i < num_elements; ++i) {
        require(atomic_numbers[i] >= 1 and atomic_numbers[i] <= 118 and (i == 0 or atomic_numbers[i - 1] < atomic_numbers[i]),
                "atomic_numbers", "expected sorted unique atomic numbers from 1 to 118");
        require(source_element_indices[i] < source_num_elements and indices.insert(source_element_indices[i]).second,
                "source_element_indices", "expected unique indices within the source species table");
    }
    atomic_energies = numbers(field(file, "atomic_energies"), "atomic_energies", num_elements);
    formal_charges = numbers(field(file, "formal_charges"), "formal_charges", num_elements);
    node_embedding = matrix(field(file, "node_embedding"), "node_embedding", num_elements, C);
    head = text(field(file, "head"), "head");
    require(not head.empty(), "head", "expected a single named head");

    const auto& architecture = field(file, "architecture");
    num_interactions = integer(field(architecture, "num_interactions", "architecture"), "architecture.num_interactions");
    require(num_interactions == 2, "architecture.num_interactions", "requires two layers");
    expect(field(architecture, "hidden_irreps", "architecture"), "architecture.hidden_irreps", "128x0e+128x1o");
    const auto& radial_info = field(architecture, "radial_embedding", "architecture");
    expect(field(radial_info, "basis", "architecture.radial_embedding"), "architecture.radial_embedding.basis", "BesselBasis");
    expect(field(radial_info, "cutoff", "architecture.radial_embedding"), "architecture.radial_embedding.cutoff", "PolynomialCutoff");
    require(integer(field(radial_info, "num_basis", "architecture.radial_embedding"), "architecture.radial_embedding.num_basis") == 8
            and integer(field(radial_info, "p", "architecture.radial_embedding"), "architecture.radial_embedding.p") == 5,
            "architecture.radial_embedding", "requires eight Bessel functions and cutoff p=5");
    const auto& layers = field(architecture, "layers", "architecture");
    require(length(layers, "architecture.layers") == 2, "architecture.layers", "requires two layers");
    const auto& mixer = field(file, "oxidation_state_mixer");
    expect(field(mixer, "type", "oxidation_state_mixer"), "oxidation_state_mixer.type", "SumOxidationStateMixer");
    parameters->embedding = oxidation(field(mixer, "embedding", "oxidation_state_mixer"), "oxidation_state_mixer.embedding");
    oxidation_state_ranges.push_back(parameters->embedding.range);
    oxidation_state_factors.push_back(parameters->embedding.factor);
    const auto& sources = field(file, "sources");
    require(length(sources, "sources") == 2, "sources", "requires two charge sources");

    for (int t = 0; t < 2; ++t) {
        const auto suffix = "_" + std::to_string(t);
        const auto layer_path = "architecture.layers[" + std::to_string(t) + "]";
        const auto& info = layers[t];
        expect(field(info, "interaction", layer_path), layer_path + ".interaction", "RealAgnosticResidualInteractionBlock");
        expect(field(info, "input_irreps", layer_path), layer_path + ".input_irreps", t == 0 ? "128x0e" : "128x0e+128x1o");
        expect(field(info, "target_irreps", layer_path), layer_path + ".target_irreps", "128x0e+128x1o+128x2e+128x3o");
        expect(field(info, "output_irreps", layer_path), layer_path + ".output_irreps", "128x0e+128x1o");
        expect(field(info, "energy_readout", layer_path), layer_path + ".energy_readout", "LinearReadoutBlock");
        require(integer(field(info, "correlation", layer_path), layer_path + ".correlation") == 3, layer_path + ".correlation", "requires order three");
        number(field(info, "avg_num_neighbors", layer_path), layer_path + ".avg_num_neighbors", true);
        auto& layer = parameters->layers[t];
        const std::vector<int> input = t == 0 ? std::vector<int>{C} : hidden;
        const std::vector<int> mid = t == 0 ? target : std::vector<int>{2 * C, 3 * C, 3 * C, 2 * C};
        const int outputs = t == 0 ? 4 * C : 10 * C;
        layer.up = linear(field(file, "linear_up" + suffix), "linear_up" + suffix, input, input);
        layer.conv = convolution(field(file, "conv_tp" + suffix), "conv_tp" + suffix, input, mid, outputs);
        layer.output = linear(field(file, "linear" + suffix), "linear" + suffix, mid, target);
        const auto& skip = field(file, "skip" + suffix);
        require(length(skip, "skip" + suffix) == num_elements, "skip" + suffix, "incorrect species count");
        for (int e = 0; e < num_elements; ++e)
            layer.skip.push_back(linear(skip[e], "skip" + suffix + "[" + std::to_string(e) + "]", t == 0 ? std::vector<int>{C, 0} : hidden, hidden));
        layer.product = product(file, t, num_elements);
        layer.product_linear = linear(field(file, "product_linear" + suffix), "product_linear" + suffix, hidden, hidden);
        layer.readout = linear(field(file, "readout" + suffix), "readout" + suffix, {C}, {1});
        const auto& values = field(file, "radial_spline_values" + suffix);
        const auto& derivs = field(file, "radial_spline_derivs" + suffix);
        require(length(values, "radial_spline_values" + suffix) == 1 and length(derivs, "radial_spline_derivs" + suffix) == 1,
                "radial_spline" + suffix, "requires one shared table");
        layer.radial = radial(values[0], derivs[0], field(file, "radial_weights" + suffix), "radial_spline" + suffix,
                              knots, outputs, radial_spline_r_min, r_cut);
        interaction_radial_shapes.push_back({layer.radial.functions, knots, outputs});

        const auto path = "sources[" + std::to_string(t) + "]";
        const auto& j = sources[t];
        expect(field(j, "type", path), path + ".type", "OxidationDependentSymmetricPredictionSourceBlock");
        expect(field(j, "multipole_irreps", path), path + ".multipole_irreps", "1x0e+1x1o");
        require(integers(field(j, "radial_mlp", path), path + ".radial_mlp", 5) == std::vector<int>({8, 32, 32, 32, 256})
                and integer(field(j, "num_radial_weights", path), path + ".num_radial_weights") == 256,
                path + ".radial_mlp", "requires 8/32/32/32/256");
        require(boolean(field(j, "zero_charges", path), path + ".zero_charges"), path + ".zero_charges", "local monopoles must be replaced by transfers");
        auto& source = parameters->sources[t];
        source.embedding = oxidation(field(j, "oxidation_embedding", path), path + ".oxidation_embedding");
        oxidation_state_ranges.push_back(source.embedding.range);
        oxidation_state_factors.push_back(source.embedding.factor);
        source.modulation.num_channels = C;
        source.modulation.weights = matrix(field(j, "oxidation", path), path + ".oxidation", 2, C * C);
        source.modulation.weights_trans = source.modulation.weights;
        for (int l = 0; l < 2; ++l)
            for (int a = 0; a < C; ++a)
                for (int b = 0; b < C; ++b)
                    source.modulation.weights_trans[l][b * C + a] = source.modulation.weights[l][a * C + b];
        source.linear1 = linear(field(j, "linear1", path), path + ".linear1", hidden, hidden);
        source.linear2 = linear(field(j, "linear2", path), path + ".linear2", hidden, hidden);
        source.conv = convolution(field(j, "conv_tp", path), path + ".conv_tp", hidden, {C}, 2 * C);
        source.readout = linear(field(j, "transfer_readout", path), path + ".transfer_readout", {C}, {1});
        source.multipoles = linear(field(j, "multipoles", path), path + ".multipoles", hidden, {1, 1});
        source.transfer_scale = number(field(j, "transfer_scale", path), path + ".transfer_scale", true);
        require(source.transfer_scale == 1.0 / 40.0, path + ".transfer_scale", "expected 1/40");
        source.radial = radial(field(j, "radial_spline_values", path), field(j, "radial_spline_derivs", path), field(j, "radial_weights", path),
                               path + ".radial_spline", knots, 2 * C, radial_spline_r_min, r_cut);
        require(source.radial.functions == 32, path + ".radial_spline", "requires 32 hidden functions");
        source_radial_shapes.push_back({source.radial.functions, knots, 2 * C});
    }
    kspace_cutoff = number(field(file, "kspace_cutoff"), "kspace_cutoff", true);
    const auto& electrostatics = field(file, "electrostatics");
    require(integer(field(electrostatics, "density_max_l", "electrostatics"), "electrostatics.density_max_l") == 1,
            "electrostatics.density_max_l", "requires charge and dipole density");
    density_smearing_width = number(field(electrostatics, "density_smearing_width", "electrostatics"), "electrostatics.density_smearing_width", true);
    include_self_interaction = boolean(field(electrostatics, "include_self_interaction", "electrostatics"), "electrostatics.include_self_interaction");
    default_pbc_handling = text(field(electrostatics, "default_pbc_handling", "electrostatics"), "electrostatics.default_pbc_handling");
    const std::set<std::string> modes = {"realspace", "pbc", "slab", "molecule_in_box", "mixed_periodic", "auto"};
    require(modes.contains(default_pbc_handling), "electrostatics.default_pbc_handling", "unknown boundary mode");
    // Older exports remain usable for local/source tests, but cannot silently
    // invent the constants needed for an electrostatic energy evaluation.
    if (electrostatics.contains("energy_parameters"))
        parameters->electrostatics = std::make_unique<Electrostatics>(energy_parameters(electrostatics.at("energy_parameters")));
    const auto& order = field(electrostatics, "density_coefficient_order", "electrostatics");
    require(order == json({"q", "dipole_y", "dipole_z", "dipole_x"}), "electrostatics.density_coefficient_order", "unsupported density component order");
}

void MACELSC::publish_forward_features(int layer)
{
    const auto& graph=graph_layout();const auto& owned=layer==1?H1:H2;
    if(graph.num_ghosts()){
        auto& halo=parameters->halo_features[layer-1];halo.assign(long(graph.num_nodes())*4*C,0.0);
        for(int i=0;i<graph.num_nodes();++i){const int owner=graph.local_owners()[i];
            if(owner>=0)std::copy_n(owned.data()+long(owner)*4*C,4*C,halo.data()+long(i)*4*C);}
    }
    parameters->forward.publish(layer,graph);
}
std::span<const double> MACELSC::forward_features(int layer) const
{
    parameters->forward.check_available(layer);
    if(graph_layout().num_ghosts())return parameters->halo_features[layer-1];
    return layer==1?std::span<const double>(H1):std::span<const double>(H2);
}
void MACELSC::pack_forward_features(int layer,std::span<const int> slots,std::span<double> values) const
{
    parameters->forward.check_pack(layer,slots,graph_layout());
    require(values.size()==slots.size()*4*C,"forward","incorrect send buffer size");
    const auto h=forward_features(layer);
    for(size_t i=0;i<slots.size();++i)std::copy_n(h.data()+long(slots[i])*4*C,4*C,values.data()+long(i)*4*C);
}
void MACELSC::unpack_forward_features(int layer,int first,std::span<const double> values)
{
    require(parameters->local_stage==layer,"forward","receive features before advancing to the next stage");
    parameters->forward.check_receive(layer,first,values,graph_layout(),4*C,std::numeric_limits<double>::max());
    auto& halo=parameters->halo_features[layer-1];
    if(!values.empty())std::copy(values.begin(),values.end(),halo.data()+long(first)*4*C);
    parameters->forward.received(layer,first,values.size()/(4*C));
}
void MACELSC::complete_forward_exchange(int layer)
{
    parameters->forward.complete(layer,graph_layout());
}
bool MACELSC::forward_exchange_complete(int layer) const
{
    return parameters->forward.complete(layer);
}

void MACELSC::compute_partial_charges(std::span<const double> positions)
{
    require(has_local_results,"stages","compute readouts before partial charges");
    const auto& graph=graph_layout();const int n=graph.num_owned(),all=graph.num_nodes();
    require(positions.size()==size_t(n)*3,"positions","expected owned positions[N_owned,3]");
    for(double x:positions)require(finite_value(x),"positions","expected finite coordinates");
    clear_charge_results();
    if(!graph.uses_remote_ghosts()){
        // Preserve the validated serial source arithmetic, including local image folding.
        evaluate_charges(positions);
        for(int t=0;t<2;++t){auto& rho=parameters->partial_layer_density[t];rho.assign(size_t(all)*4,0.0);
            std::copy(layer_density_coefficients[t].begin(),layer_density_coefficients[t].end(),rho.begin());
            parameters->all_source_features[t]=source_features[t];}
    }else{
        parameters->positions.assign(positions.begin(),positions.end());
        const LSCSlotAdjacency adjacency(graph);const auto senders=graph.feature_senders();
        edge_fluxes.assign(graph.num_edges(),0.0);
        for(int t=0;t<2;++t){auto& source=parameters->sources[t];
            if(graph.num_edges())prepare_radial(source.radial,radial_spline_r_min,r_cut);
            const auto h=forward_features(t+1);const std::vector<double> hidden(h.begin(),h.end());
            compute_source(source,graph.types(),senders,graph.receivers(),formal_charges,hidden,
                parameters->radii,parameters->Y,adjacency.incoming_starts,adjacency.incoming_edges,
                adjacency.outgoing_starts,adjacency.outgoing_edges,parameters->all_source_features[t],
                parameters->partial_layer_density[t],layer_edge_fluxes[t]);
            for(int e=0;e<graph.num_edges();++e)edge_fluxes[e]+=layer_edge_fluxes[t][e];
        }
    }
    parameters->charge_exchange.begin(graph);has_charge_results=false;
}
std::vector<double> MACELSC::download_partial_charges() const
{
    LSCChargeExchange::require(parameters->charge_exchange.partial(),"compute partial charges first");
    const int all=graph_layout().num_nodes();std::vector<double> out(size_t(all)*2);
    for(int i=0;i<all;++i)for(int t=0;t<2;++t)out[2*i+t]=parameters->partial_layer_density[t][4*i];
    return out;
}
void MACELSC::pack_reverse_charges(int first,int count,std::span<double> values)
{
    parameters->charge_exchange.check_pack(first,count,graph_layout());
    LSCChargeExchange::require(values.size()==size_t(count)*2,"incorrect reverse charge send buffer size");
    for(int i=0;i<count;++i)for(int t=0;t<2;++t){double& q=parameters->partial_layer_density[t][4*(first+i)];values[2*i+t]=q;q=0;}
    parameters->charge_exchange.sent(first,count);
}
void MACELSC::unpack_reverse_charges(std::span<const int> slots,std::span<const double> values)
{
    parameters->charge_exchange.check_add(slots,values,graph_layout());
    for(size_t i=0;i<slots.size();++i)for(int t=0;t<2;++t){
        double& q=parameters->partial_layer_density[t][4*slots[i]];q+=values[2*i+t];
        require(finite_value(q),"partial_charges","nonfinite accumulated charge");}
}
void MACELSC::complete_charge_exchange()
{
    parameters->charge_exchange.finish(graph_layout());
    const auto& graph=graph_layout();const auto node_types=graph.owned_types();
    const auto& positions=parameters->positions;const auto& edge_vectors=graph.edge_vectors();
    const int n=graph.num_owned(),count=graph.num_edges();
    density_coefficients.assign(size_t(n)*4,0.0);
    for(int i=0;i<n;++i)density_coefficients[4*i]=formal_charges[node_types[i]];
    for(int t=0;t<2;++t){
        layer_density_coefficients[t].assign(parameters->partial_layer_density[t].begin(),parameters->partial_layer_density[t].begin()+size_t(n)*4);
        source_features[t].assign(parameters->all_source_features[t].begin(),parameters->all_source_features[t].begin()+size_t(n)*4*C);
        for(size_t k=0;k<density_coefficients.size();++k)density_coefficients[k]+=layer_density_coefficients[t][k];
    }
    partial_charges.resize(n);partial_dipoles.resize(size_t(n)*3);
    dipole={};formal_charge_dipole={};transfer_dipole={};atomic_dipole_sum={};
    constexpr int cartesian[3]={3,1,2};
    for(int i=0;i<n;++i){partial_charges[i]=density_coefficients[4*i];
        for(int d=0;d<3;++d){double value=density_coefficients[4*i+cartesian[d]];
            partial_dipoles[3*i+d]=value;atomic_dipole_sum[d]+=value;
            formal_charge_dipole[d]+=formal_charges[node_types[i]]*positions[3*i+d];}}
    for(int e=0;e<count;++e)for(int d=0;d<3;++d)transfer_dipole[d]+=edge_fluxes[e]*edge_vectors[3*e+d];
    for(int d=0;d<3;++d)dipole[d]=formal_charge_dipole[d]+transfer_dipole[d]+atomic_dipole_sum[d];
    for(double x:density_coefficients)require(finite_value(x),"density_coefficients","nonfinite prediction");
    has_charge_results=true;
}
void MACELSC::set_owned_potentials(std::span<const double> values)
{
    clear_force_results();has_energy_results=false;
    LSCChargeExchange::require(has_charge_results,"assemble owned charges before setting potentials");
    parameters->charge_exchange.set_potentials(graph_layout(),values);
    const auto& graph=graph_layout();auto& halo=parameters->potential_halo;halo.assign(size_t(graph.num_nodes())*4,0.0);
    for(int i=0;i<graph.num_nodes();++i){int owner=graph.local_owners()[i];if(owner>=0)
        std::copy_n(values.data()+4*owner,4,halo.data()+4*i);}
}
std::span<const double> MACELSC::electrostatic_potentials() const
{
    LSCChargeExchange::require(parameters->charge_exchange.potentials_available(),"set owned electrostatic potentials first");
    return parameters->potential_halo;
}
void MACELSC::pack_potentials(std::span<const int> slots,std::span<double> values) const
{
    parameters->charge_exchange.check_potential_pack(slots,graph_layout());
    LSCChargeExchange::require(values.size()==4*slots.size(),"incorrect potential send buffer size");
    for(size_t i=0;i<slots.size();++i)std::copy_n(parameters->potential_halo.data()+4*slots[i],4,values.data()+4*i);
}
void MACELSC::unpack_potentials(int first,std::span<const double> values)
{
    parameters->charge_exchange.check_potential_receive(first,values,graph_layout());
    if(!values.empty())std::copy(values.begin(),values.end(),parameters->potential_halo.data()+4*first);
    parameters->charge_exchange.received_potentials(first,values.size()/4);
}
void MACELSC::complete_potential_exchange(){parameters->charge_exchange.complete_potentials(graph_layout());}
bool MACELSC::potential_exchange_complete()const{return parameters->charge_exchange.potentials_ready();}
std::string MACELSC::charge_exchange_stage()const{return parameters->charge_exchange.stage();}

int MACELSC::distributed_reverse_phase()const{return parameters->reverse_exchange.phase;}
void MACELSC::begin_distributed_reverse(std::span<const double> pg,const std::array<double,9>& es,
    const std::array<double,9>& cell,const std::array<double,3>& field,double el)
{
    const auto& g=graph_layout();const int N=g.num_owned(),M=g.num_nodes(),E=g.num_edges();
    LSCReverseExchange::check(parameters->charge_exchange.potentials_ready(),"complete potential exchange first");
    LSCReverseExchange::check(pg.size()==size_t(N)*3,"expected owned position gradient[N_owned,3]");
    LSCChargeExchange::finite(pg);LSCChargeExchange::finite(es);LSCChargeExchange::finite(cell);LSCChargeExchange::finite(field);LSCChargeExchange::finite({&el,1});
    clear_force_results();parameters->energy_cell=cell;parameters->energy_field=field;
    electrostatic_energy=el;field_energy=0;for(int d=0;d<3;++d)field_energy+=field[d]*dipole[d];
    energy=local_energy+el+field_energy;has_energy_results=true;
    strain_derivative=electrostatic_strain_derivative=es;
    parameters->position_adj.assign(pg.begin(),pg.end());parameters->edge_adj.assign(size_t(E)*3,0);
    parameters->flux_adj.assign(E,0);auto density=parameters->potential_halo;
    constexpr int cart[3]={3,1,2};
    for(int i=0;i<M;++i)for(int d=0;d<3;++d){
        if(i>=N)density[4*i+cart[d]]=0;
        else{density[4*i+cart[d]]+=field[d];double x=formal_charges[g.types()[i]]*field[d];
            parameters->position_adj[3*i+d]+=x;for(int a=0;a<3;++a)strain_derivative[3*a+d]+=parameters->positions[3*i+a]*x;}}
    density_energy_gradient.assign(density.begin(),density.begin()+size_t(N)*4);
    for(int e=0;e<E;++e)for(int d=0;d<3;++d){parameters->flux_adj[e]+=field[d]*g.edge_vectors()[3*e+d];parameters->edge_adj[3*e+d]+=edge_fluxes[e]*field[d];}
    parameters->Y_grad.resize(size_t(E)*48);
    if(E){sphericart::SphericalHarmonics<double> harmonics(3);harmonics.compute_array_with_gradients(g.edge_vectors().data(),g.edge_vectors().size(),parameters->Y.data(),parameters->Y.size(),parameters->Y_grad.data(),parameters->Y_grad.size());}
    const bool remote=g.uses_remote_ghosts();const int K=remote?M:N;LSCSlotAdjacency adj(g);const auto senders=g.feature_senders();
    for(int t=0;t<2;++t){auto& gradient=parameters->mpi_H_adj[t];gradient.assign(size_t(M)*4*C,0);
        if(N){std::vector<double> ones(N,1);parameters->layers[t].readout.reverse(N,ones.data(),1,gradient.data(),4*C);}
        if(K){std::vector<double> H=remote?parameters->halo_features[t]:(t==0?H1:H2);
            std::vector<double> da(density.begin(),density.begin()+size_t(K)*4);
            reverse_source(parameters->sources[t],senders,g.receivers(),adj.incoming_starts,adj.incoming_edges,adj.outgoing_starts,adj.outgoing_edges,
                parameters->radii,parameters->Y,parameters->Y_grad,g.edge_vectors(),H,da,parameters->flux_adj,gradient,parameters->edge_adj);}}
    parameters->reverse_exchange.begin(g);
}
void MACELSC::pack_reverse_gradients(int layer,int first,int count,std::span<double> values)
{
    LSCReverseExchange::check(layer==1 || layer==2,"expected layer 1 or 2");auto& state=parameters->reverse_exchange;
    state.check_pack(layer==2?1:3,first,count,4*C,values,graph_layout());auto& x=parameters->mpi_H_adj[layer-1];
    if(!values.empty()){std::copy_n(x.data()+size_t(first)*4*C,values.size(),values.data());std::fill_n(x.data()+size_t(first)*4*C,values.size(),0);}state.sent(first,count);
}
void MACELSC::unpack_reverse_gradients(int layer,std::span<const int> slots,std::span<const double> values)
{
    LSCReverseExchange::check(layer==1 || layer==2,"expected layer 1 or 2");parameters->reverse_exchange.check_add(layer==2?1:3,slots,4*C,values,graph_layout());
    auto& x=parameters->mpi_H_adj[layer-1];for(size_t k=0;k<slots.size();++k)for(int d=0;d<4*C;++d)x[size_t(slots[k])*4*C+d]+=values[k*4*C+d];
}
void MACELSC::complete_reverse_gradient_exchange(int layer){LSCReverseExchange::check(layer==1 || layer==2,"expected layer 1 or 2");parameters->reverse_exchange.finish(layer==2?1:3,graph_layout());}
void MACELSC::reverse_distributed_second_layer()
{
    auto& state=parameters->reverse_exchange;state.require_phase(2);const auto& g=graph_layout();LSCSlotAdjacency adj(g);const int N=g.num_owned(),K=g.uses_remote_ghosts()?g.num_nodes():N;
    if(N){std::vector<double> owned(parameters->mpi_H_adj[1].begin(),parameters->mpi_H_adj[1].begin()+size_t(N)*4*C);
        reverse_layer(parameters->layers[1],g.owned_types(),g.receivers(),adj.outgoing_starts,adj.outgoing_edges,parameters->radii,parameters->Y,parameters->Y_grad,g.edge_vectors(),A1,owned,parameters->mpi_H_adj[0],parameters->edge_adj,K);}
    state.next(2,3,g);
}
void MACELSC::reverse_distributed_first_layer()
{
    auto& state=parameters->reverse_exchange;state.require_phase(4);const auto& g=graph_layout();LSCSlotAdjacency adj(g);const int N=g.num_owned(),K=g.uses_remote_ghosts()?g.num_nodes():N;
    if(N){std::vector<double> owned(parameters->mpi_H_adj[0].begin(),parameters->mpi_H_adj[0].begin()+size_t(N)*4*C),initial(size_t(K)*C,0);
        reverse_layer(parameters->layers[0],g.owned_types(),g.receivers(),adj.outgoing_starts,adj.outgoing_edges,parameters->radii,parameters->Y,parameters->Y_grad,g.edge_vectors(),A0,owned,initial,parameters->edge_adj,K);}
    state.next(4,5,g);
}
void MACELSC::finalize_distributed_forces(bool want_stress)
{
    auto& state=parameters->reverse_exchange;state.require_phase(5);const auto& g=graph_layout();const int N=g.num_owned(),M=g.num_nodes(),E=g.num_edges();
    const auto senders=g.feature_senders();auto& all=parameters->mpi_forces;all.assign(size_t(M)*3,0);position_forces.resize(size_t(N)*3);edge_forces.resize(size_t(E)*3);
    for(size_t k=0;k<size_t(N)*3;++k)all[k]=position_forces[k]=-parameters->position_adj[k];
    for(int e=0;e<E;++e)for(int d=0;d<3;++d){double x=parameters->edge_adj[3*e+d];edge_forces[3*e+d]=-x;all[3*g.receivers()[e]+d]-=x;all[3*senders[e]+d]+=x;
        for(int a=0;a<3;++a)strain_derivative[3*a+d]+=g.edge_vectors()[3*e+a]*x;}
    const auto w=strain_derivative;for(int a=0;a<3;++a)for(int b=0;b<3;++b)strain_derivative[3*a+b]=.5*(w[3*a+b]+w[3*b+a]);
    const auto& c=parameters->energy_cell;double volume=std::abs(c[0]*(c[4]*c[8]-c[5]*c[7])-c[1]*(c[3]*c[8]-c[5]*c[6])+c[2]*(c[3]*c[7]-c[4]*c[6]));
    if(want_stress)LSCReverseExchange::check(finite_value(volume) && volume>0,"stress requires nonsingular cell");
    constexpr int po[6]={0,4,8,1,2,5},ao[6]={0,4,8,5,2,1};for(int d=0;d<6;++d){virial[d]=-strain_derivative[po[d]];if(want_stress)stress[d]=strain_derivative[ao[d]]/volume;}
    LSCChargeExchange::finite(all);LSCChargeExchange::finite(strain_derivative);has_stress_results=want_stress;state.next(5,6,g);
}
void MACELSC::pack_reverse_forces(int first,int count,std::span<double> values){auto& s=parameters->reverse_exchange;s.check_pack(6,first,count,3,values,graph_layout());auto& f=parameters->mpi_forces;if(!values.empty()){std::copy_n(f.data()+size_t(first)*3,values.size(),values.data());std::fill_n(f.data()+size_t(first)*3,values.size(),0);}s.sent(first,count);}
void MACELSC::unpack_reverse_forces(std::span<const int> slots,std::span<const double> values){parameters->reverse_exchange.check_add(6,slots,3,values,graph_layout());for(size_t k=0;k<slots.size();++k)for(int d=0;d<3;++d)parameters->mpi_forces[3*slots[k]+d]+=values[3*k+d];}
void MACELSC::complete_force_exchange(){parameters->reverse_exchange.finish(6,graph_layout());forces.assign(parameters->mpi_forces.begin(),parameters->mpi_forces.begin()+size_t(graph_layout().num_owned())*3);has_force_results=true;}
