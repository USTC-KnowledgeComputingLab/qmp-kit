#include <pybind11/complex.h>
#include <torch/extension.h>

namespace qmp_hamiltonian {

// The `prepare` function is responsible for parsing a raw Python dictionary representing Hamiltonian terms
// and transforming it into a structured tuple of tensors. This tuple is then stored on the Python side
// and utilized in subsequent calls to the PyTorch operators for further processing.
//
// The function takes a Python dictionary `hamiltonian` as input, where each key-value pair represents a term
// in the Hamiltonian. The key is a tuple of tuples, where each inner tuple contains two elements:
// - The first element is an integer representing the site index of the operator.
// - The second element is an integer representing the type of operator (0 for annihilation, 1 for creation).
// The value is either a float or a complex number representing the coefficient of the term.
//
// The function processes the dictionary and constructs three tensors:
// - `site`: An int16 tensor of shape [term_number, max_op_number], representing the site indices of the operators for
//   each term.
// - `kind`: An uint8 tensor of shape [term_number, max_op_number], representing the type of operator for each term.
//   The value are encoded as follows:
//   - 0: Annihilation operator
//   - 1: Creation operator
//   - 2: Empty (identity operator)
// - `coef`: A float64 tensor of shape [term_number, 2], representing the coefficients of each term, with two elements
//   for real and imaginary parts.
//
// The `max_op_number` template argument specifies the maximum number of operators per term, typically set to 4 for
// 2-body interactions.
template<std::int64_t max_op_number>
auto prepare(py::dict hamiltonian) {
    std::int64_t term_number = hamiltonian.size();

    auto site = torch::empty({term_number, max_op_number}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    // No need to initialize
    auto kind = torch::full({term_number, max_op_number}, 2, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    // Initialize to 2 for identity as default
    auto coef = torch::empty({term_number, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    // No need to initialize

    auto site_accessor = site.accessor<std::int16_t, 2>();
    auto kind_accessor = kind.accessor<std::uint8_t, 2>();
    auto coef_accessor = coef.accessor<double, 2>();

    std::int64_t index = 0;
    for (auto& item : hamiltonian) {
        auto key = item.first.cast<py::tuple>();
        auto value_is_float = py::isinstance<py::float_>(item.second);
        auto value = value_is_float ? std::complex<double>(item.second.cast<double>()) : item.second.cast<std::complex<double>>();

        std::int64_t op_number = key.size();
        for (std::int64_t i = 0; i < op_number; ++i) {
            auto tuple = key[i].cast<py::tuple>();
            site_accessor[index][i] = tuple[0].cast<std::int16_t>();
            kind_accessor[index][i] = tuple[1].cast<std::uint8_t>();
        }

        coef_accessor[index][0] = value.real();
        coef_accessor[index][1] = value.imag();

        ++index;
    }

    return std::make_tuple(site, kind, coef);
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES == 0
// Expose the `prepare` function to Python.
PYBIND11_MODULE(qmp_hamiltonian, m) {
    m.def("prepare", prepare</*max_op_number=*/4>, py::arg("hamiltonian"));
}
#endif

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(x, y) qmp_hamiltonian_##x##_##y
#define QMP_LIBRARY(x, y) QMP_LIBRARY_HELPER(x, y)
TORCH_LIBRARY_FRAGMENT(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT), m) {
    m.def("apply_within(Tensor configs_i, Tensor psi_i, Tensor configs_j, Tensor site, Tensor kind, Tensor coef) -> Tensor");
    m.def("find_relative(Tensor configs_i, Tensor psi_i, int count_selected, Tensor site, Tensor kind, Tensor coef, Tensor configs_exclude) -> Tensor"
    );
    m.def("diagonal_term(Tensor configs, Tensor site, Tensor kind, Tensor coef) -> Tensor");
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif

} // namespace qmp_hamiltonian
