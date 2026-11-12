#include <incompressible_chns_solver.h>

#include "../tests.h"

#include "presolver_test_parameters.h"

struct CheckedBand
{};
class BandProblem : public CHNSSolver<2, false>
{
public:
  BandProblem(const ParameterReader<2> &p, double margin)
    : CHNSSolver<2, false>(p)
    , margin(margin)
  {
    pcout.set_condition(false);
  }
  void solver_specific_post_processing() override
  {
    compute_error_estimate();
    std::array<unsigned int, 3> counts{};
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        double lo = 1., hi = 0.;
        for (const auto &v : cell->vertex_indices())
        {
          lo = std::min(lo, cell->vertex(v)[0]);
          hi = std::max(hi, cell->vertex(v)[0]);
        }
        const double distance =
          lo <= .5 && hi >= .5 ? 0. :
                                 std::min(std::abs(lo - .5), std::abs(hi - .5));
        const double inner =
          std::sqrt(2.) * .05 * std::atanh(.9) + margin * .05;
        const double outer =
          std::sqrt(2.) * .05 * std::atanh(.999) + margin * .05;
        const unsigned expected = distance < inner ? 4 :
                                  distance < outer ? 3 :
                                                     1;
        AssertThrow(cellwise_refinement_criterion[cell->active_cell_index()] ==
                      expected,
                    ExcMessage("Incorrect interface-band target level"));
        ++counts[expected == 4 ? 0 : expected == 3 ? 1 : 2];
      }
    for (const auto count : counts)
      AssertThrow(Utilities::MPI::sum(count, mpi_communicator) > 0,
                  ExcMessage("Test must cover inner, outer and far bands"));
    throw CheckedBand{};
  }

private:
  double margin;
};
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    for (double margin : {0., 1.})
    {
      std::ostringstream extra;
      extra << R"(subsection Cahn Hilliard
 set use presolver = false
 set interface thickness = 0.05
end
subsection Mesh
 set dealii mesh parameters = 8,2:0,0:1,1:true
 subsection Adaptation
  set strategy = local refinement
  subsection Local refinement
   set variables for adaptation = phase_tracer
   set refinement strategy = interface band
   set interface band thresholds = 0.9, 0.999
   set min grid level = 1
   set max grid level = 4
   set interface band margin = )"
            << margin << R"(
  end
 end
end
subsection Initial conditions
 subsection cahn hilliard tracer
  set Function expression = tanh((x-0.5)/(sqrt(2)*0.05))
 end
end
)";
      auto p       = presolver_test_parameters<2>(extra.str());
      bool checked = false;
      try
      {
        BandProblem(p, margin).run();
      }
      catch (const CheckedBand &)
      {
        checked = true;
      }
      AssertThrow(checked, ExcMessage("Initial band test did not run"));
    }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      deallog << "Interface-band levels and margin OK" << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
