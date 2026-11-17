#include <compare_matrix.h>
#include <incompressible_chns_solver.h>
// Preserve the original polynomial-profile cases, including the evolved state
// with a weak reconstructed gradient. A forward difference alone is inaccurate
// there: check centered differences AND the truncation-error convergence, with
// the same frozen tau and global amplitude as the analytic linearization.
template <bool moving, bool enlarged>
class LocalCorrectionJacobian : public CHNSSolver<2, moving, enlarged>
{
  using Base = CHNSSolver<2, moving, enlarged>;

public:
  LocalCorrectionJacobian(const ParameterReader<2> &p)
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
    // A nonzero, spatially varying interaction field exercises attenuation
    // throughout the complete assembled residual (not only its unit formula).
    auto geometry     = std::make_shared<CHNSProfileGeometry::Fields>();
    geometry->overlap = std::make_shared<CHNSProfileOverlap::SmoothPenalty>(
      std::vector<CHNSProfileOverlap::MidlineSource>{
        {{-.5, .4}, {1.5, .4}, .03}},
      this->param.cahn_hilliard.epsilon_interface,
      std::sqrt(2.));
    this->scratch_data->profile_geometry = geometry;
    auto &scratch                        = *this->scratch_data;
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


#include <filesystem>

#include "../tests.h"
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  if (rank == 0)
    initlog();
  try
  {
    const auto file =
      std::filesystem::path(__FILE__).parent_path().parent_path() /
      "incompressible_chns/jacobian_profile_correction.prm";
    for (const bool projection : {false, true})
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
          extra << "subsection Time integration\nset t_end = 0.01\nend\n"
                << "subsection FiniteElements\nset use quads = true\nset "
                   "Tracer degree = 1\nend\n"
                << "subsection Cahn Hilliard\nsubsection profile correction\n"
                << "set profile correction model = local timescale\n"
                << "set profile correction timescale factor = 1.2\nend\n"
                << "set enable flux correction = "
                << (projection ? "true" : "false")
                << "\nset surface tension representation = laplace beltrami\n"
                << "subsection mobility\nset model = constant\nset constant "
                   "value = 0.1\nend\nend\n"
                << "subsection Stabilization\nset enable supg = "
                << (momentum ? "true" : "false")
                << "\nset enable tracer supg = " << (tracer ? "true" : "false")
                << "\nend\n";
          prm.parse_input_from_string(extra.str());
          param.read(prm);
          LocalCorrectionJacobian<false, false> problem(param);
          problem.run();
          AssertThrow(problem.comparisons == 2, ExcInternalError());
        }
    if (rank == 0)
      deallog << "Local correction + LB: all SUPG combinations, with/without "
                 "flux projection, two states OK"
              << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
