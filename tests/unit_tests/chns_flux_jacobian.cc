#include <compare_matrix.h>
#include <incompressible_chns_solver.h>

#include <filesystem>

#include "../tests.h"

#include "chns_jacobian_sweep.h"

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
      CHNSJacobianSweepProblem<moving, enlarged> problem(param);
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
