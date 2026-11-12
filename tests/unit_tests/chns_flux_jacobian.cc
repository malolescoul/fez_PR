#include <compare_matrix.h>
#include <incompressible_chns_solver.h>

#include <filesystem>

#include "../tests.h"

// Preserve the original polynomial-profile cases, including the evolved state
// with a weak reconstructed gradient. A forward difference alone is inaccurate
// there: check centered differences AND the truncation-error convergence, with
// the same frozen tau and global amplitude as the analytic linearization.
template <bool moving, bool enlarged>
class FluxJacobianProblem : public CHNSSolver<2, moving, enlarged>
{
  using Base = CHNSSolver<2, moving, enlarged>;

public:
  FluxJacobianProblem(const ParameterReader<2> &p)
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
    std::cout << "Flux Jacobian " << moving << enlarged << " step "
              << comparisons << " forward " << forward[0] << ' ' << forward[1]
              << ' ' << forward[2] << " centered " << centered[0] << ' '
              << centered[1] << ' ' << centered[2] << std::endl;
    AssertThrow(centered[2] < 2e-5,
                ExcMessage(
                  "Incorrect assembled flux Jacobian on original profile"));
    if constexpr (!moving && !enlarged)
      if (comparisons == 2 && this->param.stabilization.enable_supg &&
          this->param.stabilization.enable_tracer_supg)
      {
        // This is the original failing case: retain its sensitivity, rather
        // than replacing it with an easier field or loosening its tolerance.
        AssertThrow(
          forward[2] > 1e-4 && forward[0] / forward[1] > 8. &&
            forward[0] / forward[1] < 12. && forward[1] / forward[2] > 8. &&
            forward[1] / forward[2] < 12.,
          ExcMessage(
            "Expected first-order forward truncation was not reproduced"));
        AssertThrow(
          centered[0] / centered[1] > 70. && centered[0] / centered[1] < 130.,
          ExcMessage(
            "Expected second-order centered convergence was not reproduced"));
      }
  }
};

template <bool moving, bool enlarged>
void run_cases(const std::string &solver)
{
  const auto file =
    std::filesystem::path(__FILE__).parent_path().parent_path() / solver /
    "jacobian_profile_correction.prm";
  for (const bool momentum : {false, true})
    for (const bool tracer : {false, true})
    {
      Parameters::BoundaryConditionsData bc;
      read_number_of_boundary_conditions(file.string(), bc);
      ParameterReader<2> param(bc);
      ParameterHandler   prm;
      param.declare(prm);
      prm.parse_input(file.string());
      std::ostringstream extra;
      extra
        << "subsection Time integration\nset t_end = 0.01\nend\n"
        << "subsection Cahn Hilliard\nset enable flux correction = true\nend\n"
        << "subsection Stabilization\nset enable supg = "
        << (momentum ? "true" : "false")
        << "\nset enable tracer supg = " << (tracer ? "true" : "false")
        << "\nend\n";
      std::istringstream input(extra.str());
      prm.parse_input(input);
      param.read(prm);
      FluxJacobianProblem<moving, enlarged> problem(param);
      problem.run();
      AssertThrow(problem.comparisons == 2,
                  ExcMessage("Both time states must be checked"));
    }
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    run_cases<false, false>("incompressible_chns");
    run_cases<true, false>("incompressible_chns_ale");
    run_cases<true, true>("incompressible_chns_ale_enlarged");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      deallog << "Original-profile flux Jacobians: 3 solvers, 4 stabilization "
                 "combinations, 2 time states OK"
              << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
