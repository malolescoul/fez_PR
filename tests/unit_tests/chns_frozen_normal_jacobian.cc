#include <compare_matrix.h>
#include <incompressible_chns_solver.h>

#include <filesystem>

#include "../tests.h"

#include "chns_jacobian_sweep.h"

// Assembled CHNS Jacobian with the frozen normal of the profile correction,
// against centred differences of the residual. The normal comes from the
// previous time step, so it is fixed during the comparison; the second time
// state checks a normal that differs from the current gradient.
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    const auto file =
      std::filesystem::path(__FILE__).parent_path().parent_path() /
      "incompressible_chns" / "jacobian_profile_correction.prm";
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
              << "subsection Cahn Hilliard\nset surface tension "
                 "representation = laplace beltrami\nsubsection mobility\nset "
                 "model = constant\nset constant value = 0.1\nend\n"
              << "subsection profile correction\nset frozen normal = true\n"
                 "end\nend\n"
              << "subsection Stabilization\nset enable supg = "
              << (momentum ? "true" : "false")
              << "\nset enable tracer supg = " << (tracer ? "true" : "false")
              << "\nend\n";
        std::istringstream input(extra.str());
        prm.parse_input(input);
        param.read(prm);
        CHNSJacobianSweepProblem<false, false> problem(param);
        problem.run();
        AssertThrow(problem.comparisons == 2,
                    ExcMessage("Both time states must be checked"));
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      deallog << "Frozen-normal profile correction Jacobian: 4 stabilization "
                 "combinations, 2 time states OK"
              << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
