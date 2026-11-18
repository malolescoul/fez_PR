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

// Exercise the real distributed marking and adaptation, including conflicting
// Kelly coarsening and band protection. Synthetic Kelly scores make the
// refinement/coarsening locations independent of solver roundoff.
class CombinedBandProblem : public CHNSSolver<2, false>
{
public:
  CombinedBandProblem(const ParameterReader<2> &p, const bool zero_velocity)
    : CHNSSolver<2, false>(p)
    , zero_velocity(zero_velocity)
  {
    pcout.set_condition(false);
  }

  void solver_specific_post_processing() override
  {
    compute_error_estimate();
    const double kelly_max =
      Utilities::MPI::max(cellwise_refinement_criterion.linfty_norm(),
                          mpi_communicator);
    AssertThrow(zero_velocity ? kelly_max == 0. : kelly_max > 0.,
                ExcMessage("Combined mode must compute Kelly on velocity"));

    // Cells near the interface must retain/refine to the band level even
    // though their synthetic Kelly score requests coarsening.
    const unsigned int target = param.mesh.adaptation.tree_amr.max_level;
    for (const auto &cell : dof_handler->active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      const auto index = cell->active_cell_index();
      if (std::abs(cell->center()[0] - .5) < .08)
        AssertThrow(cellwise_interface_band_levels[index] == target,
                    ExcMessage("Missing phase band in combined mode"));
      cellwise_refinement_criterion[index] = cell->center()[0] > .875 ?
                                               1. + cell->center()[1] :
                                               .001 * cell->center()[0];
    }
    transient_fixed_point_data.adapt_meshes(cellwise_refinement_criterion,
                                            cellwise_interface_band_levels);

    unsigned int far_refined = 0, far_coarsened = 0, protected_cells = 0;
    for (const auto &cell : triangulation->active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      if (std::abs(cell->center()[0] - .5) < .08)
      {
        AssertThrow(static_cast<unsigned int>(cell->level()) >= target,
                    ExcMessage("Kelly overrode the interface-band minimum"));
        ++protected_cells;
      }
      if (cell->center()[0] > .25 && cell->center()[0] < .3)
        AssertThrow(cell->level() >= 2,
                    ExcMessage("Kelly coarsened below the outer-band level"));
      if (cell->center()[0] > .9 && cell->level() == 3)
        ++far_refined;
      if (cell->center()[0] < .1 && cell->level() == 1)
        ++far_coarsened;
    }
    AssertThrow(Utilities::MPI::sum(protected_cells, mpi_communicator) > 0 &&
                  Utilities::MPI::sum(far_refined, mpi_communicator) > 0 &&
                  Utilities::MPI::sum(far_coarsened, mpi_communicator) > 0,
                ExcMessage("Must exercise band protection, Kelly refinement "
                           "and coarsening away from the interface"));
    throw CheckedBand{};
  }

private:
  bool zero_velocity;
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
    for (const bool zero_velocity : {true, false})
    {
      auto p = presolver_test_parameters<2>(R"(
subsection Cahn Hilliard
 set use presolver = false
 set interface thickness = 0.05
end
subsection Mesh
 set refinement level = 2
 set dealii mesh parameters = 8,2:0,0:1,1:true
 subsection Adaptation
  set enable = true
  set verbosity = quiet
  set strategy = local refinement
  subsection Local refinement
   set variables for adaptation = velocity
   set refinement strategy = interface band with kelly
   set interface band thresholds = 0.9, 0.999
   set min grid level = 1
   set max grid level = 3
   set fraction to refine = 0.125
   set fraction to coarsen = 0.5
   set number of prerefinement steps = 0
  end
 end
end
subsection Initial conditions
 subsection cahn hilliard tracer
  set Function expression = tanh((x-0.5)/(sqrt(2)*0.05))
 end
 subsection velocity
  set Function expression = sin(15*x)*y*(1-y);0
 end
end
)");
      if (zero_velocity)
      {
        ParameterHandler zero_prm;
        p.initial_conditions.initial_velocity_callback->declare_parameters(
          zero_prm, 2);
        zero_prm.set("Function expression", "0;0");
        p.initial_conditions.initial_velocity_callback->parse_parameters(
          zero_prm);
      }
      bool checked = false;
      try
      {
        CombinedBandProblem(p, zero_velocity).run();
      }
      catch (const CheckedBand &)
      {
        checked = true;
      }
      AssertThrow(checked, ExcMessage("Combined band test did not run"));
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
