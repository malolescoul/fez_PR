#ifndef CHNS_JACOBIAN_SWEEP_H
#define CHNS_JACOBIAN_SWEEP_H
#include <compare_matrix.h>
#include <incompressible_chns_solver.h>
// Compare the assembled CHNS Jacobian with finite differences on the
// polynomial-profile cases, including the evolved state. A forward difference
// alone is inaccurate there: check centered differences, with the same frozen
// tau and global amplitude as the analytic linearization.
template <bool moving, bool enlarged>
class CHNSJacobianSweepProblem : public CHNSSolver<2, moving, enlarged>
{
  using Base = CHNSSolver<2, moving, enlarged>;

public:
  CHNSJacobianSweepProblem(const ParameterReader<2> &p)
    : Base(p)
  {
    this->pcout.set_condition(false);
  }

  unsigned int comparisons = 0;

  void compare_analytical_matrix_with_fd() override
  {
    ++comparisons;
    this->local_evaluation_point = this->get_present_solution();
    this->evaluation_point       = this->get_present_solution();
    auto &scratch                = *this->scratch_data;
    scratch.profile_correction_coefficient =
      this->compute_profile_correction_coefficient();
    CopyDataBase<1>             copy(*this->fe);
    const std::array<double, 3> steps{{1e-6, 1e-7, 1e-8}};
    std::array<double, 3>       forward{}, centered{};
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        this->assemble_local_matrix(cell, scratch, copy);
        const FullMatrix<double> analytic = copy.local_matrix();
        const auto               indices  = copy.dof_indices();
        scratch.set_keep_tau_constant(true);
        this->assemble_local_rhs(cell, scratch, copy);
        const Vector<double> ref = copy.local_rhs();
        for (unsigned int k = 0; k < steps.size(); ++k)
          for (unsigned int j = 0; j < indices.size(); ++j)
          {
            const double u0 = this->evaluation_point[indices[j]];
            const double h  = steps[k];
            Verification::set_local_entry(this->evaluation_point,
                                          indices[j],
                                          u0 + h);
            this->assemble_local_rhs(cell, scratch, copy);
            const Vector<double> plus = copy.local_rhs();
            Verification::set_local_entry(this->evaluation_point,
                                          indices[j],
                                          u0 - h);
            this->assemble_local_rhs(cell, scratch, copy);
            const auto &minus = copy.local_rhs();
            for (unsigned int i = 0; i < indices.size(); ++i)
            {
              forward[k] =
                std::max(forward[k],
                         std::abs(analytic(i, j) + (plus[i] - ref[i]) / h));
              centered[k] = std::max(centered[k],
                                     std::abs(analytic(i, j) +
                                              (plus[i] - minus[i]) / (2 * h)));
            }
            Verification::set_local_entry(this->evaluation_point,
                                          indices[j],
                                          u0);
          }
        scratch.set_keep_tau_constant(false);
      }
    for (unsigned int k = 0; k < steps.size(); ++k)
    {
      forward[k]  = Utilities::MPI::max(forward[k], this->mpi_communicator);
      centered[k] = Utilities::MPI::max(centered[k], this->mpi_communicator);
    }
    std::cout << "CHNS Jacobian " << moving << enlarged << " step "
              << comparisons << " forward " << forward[0] << ' ' << forward[1]
              << ' ' << forward[2] << " centered " << centered[0] << ' '
              << centered[1] << ' ' << centered[2] << std::endl;
    AssertThrow(centered[2] < 2e-5,
                ExcMessage(
                  "Incorrect assembled CHNS Jacobian on original profile"));
  }
};

#endif
