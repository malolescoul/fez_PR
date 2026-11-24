#include <chns_profile_correction_cutoff.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "../tests.h"

// Checkpointing of the cutoff chi, which carries a state from one step to the
// next through its relaxation in time. chi is attached to the triangulation
// before it is saved and read back after it is loaded: the relaxation then
// continues exactly as without the restart. A checkpoint without chi restarts
// from chi = 0, and a chi that is not needed can be read and dropped.

constexpr double epsilon = 1.;
constexpr double dt      = 2.5e-4;

// Two overlapping tanh profiles 4 eps apart: chi rises towards 1 inside.
class Filament : public Function<2>
{
public:
  double value(const Point<2> &p, const unsigned int = 0) const override
  {
    const double s = std::sqrt(2.) * epsilon;
    return std::tanh((p[1] + 2.) / s) + std::tanh((2. - p[1]) / s) - 1.;
  }
};

void make_mesh(parallel::distributed::Triangulation<2> &tria)
{
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {12, 12},
                                            Point<2>(-6., -6.),
                                            Point<2>(6., 6.));
}

// Refine around the filament, so that chi lives on a mesh with hanging nodes.
void refine(parallel::distributed::Triangulation<2> &tria)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && std::abs(cell->center()[1]) < 3.)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
}

struct Phase
{
  Phase(const parallel::distributed::Triangulation<2> &tria)
    : dof_handler(tria)
  {
    dof_handler.distribute_dofs(fe);
    const IndexSet    owned = dof_handler.locally_owned_dofs();
    LA::ParVectorType interpolated(owned, MPI_COMM_WORLD);
    VectorTools::interpolate(mapping, dof_handler, Filament(), interpolated);
    solution.reinit(owned,
                    DoFTools::extract_locally_relevant_dofs(dof_handler),
                    MPI_COMM_WORLD);
    solution = interpolated;
  }

  double update(CahnHilliard::ProfileCorrectionCutoff<2> &cutoff,
                const unsigned int                        step) const
  {
    return cutoff
      .update(dof_handler, mapping, quadrature, solution, 0, epsilon, step, dt)
      .max_chi;
  }

  const FE_Q<2>     fe{2};
  DoFHandler<2>     dof_handler;
  const MappingQ<2> mapping{1};
  const QGauss<2>   quadrature{3};
  LA::ParVectorType solution;
};

unsigned int n_attached(const std::string &prefix)
{
  std::ifstream info(prefix + ".info");
  std::string   header;
  unsigned int  version, n_procs, n_fixed, n_variable;
  std::getline(info, header);
  info >> version >> n_procs >> n_fixed >> n_variable;
  return n_fixed + n_variable;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const bool root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  if (root)
    initlog();

  CahnHilliard::InteractionCutoffParameters parameters;
  parameters.rise_time  = 1e-3;
  parameters.decay_time = 2e-3;

  // Reference run: three steps without interruption, checkpoints after two.
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  make_mesh(tria);
  refine(tria);
  const Phase                              phase(tria);
  CahnHilliard::ProfileCorrectionCutoff<2> cutoff(tria, parameters);
  phase.update(cutoff, 1);
  phase.update(cutoff, 2);
  const double saved = cutoff.largest_chi();
  AssertThrow(cutoff.prepare_serialization(),
              ExcMessage("chi carries a state with relaxation"));
  tria.save("with_chi");
  cutoff.finish_serialization();
  tria.save("without_chi");
  const double reference = phase.update(cutoff, 3);

  // Restart with chi: the relaxation continues from the saved value.
  double restored, continued;
  {
    parallel::distributed::Triangulation<2> loaded(MPI_COMM_WORLD);
    make_mesh(loaded);
    loaded.load("with_chi");
    const Phase                              restart_phase(loaded);
    CahnHilliard::ProfileCorrectionCutoff<2> restarted(loaded, parameters);
    restarted.deserialize(false);
    restored  = restarted.largest_chi();
    continued = restart_phase.update(restarted, 3);
  }
  // Restart from a checkpoint without chi: chi starts again from zero.
  double from_zero;
  {
    parallel::distributed::Triangulation<2> loaded(MPI_COMM_WORLD);
    make_mesh(loaded);
    loaded.load("without_chi");
    const Phase                              restart_phase(loaded);
    CahnHilliard::ProfileCorrectionCutoff<2> restarted(loaded, parameters);
    from_zero = restart_phase.update(restarted, 3);
  }
  // chi not needed by the restarted run: read and dropped, after which the
  // triangulation can be refined and saved again.
  {
    parallel::distributed::Triangulation<2> loaded(MPI_COMM_WORLD);
    make_mesh(loaded);
    loaded.load("with_chi");
    CahnHilliard::ProfileCorrectionCutoff<2>::discard_serialized(loaded, false);
    refine(loaded);
    loaded.save("after_discard");
  }

  AssertThrow(std::abs(restored - saved) < 1e-12 &&
                std::abs(continued - reference) < 1e-12,
              ExcMessage("chi is not restored exactly"));
  AssertThrow(from_zero < reference - 0.05,
              ExcMessage("a checkpoint without chi must restart from zero"));
  if (root)
  {
    deallog << std::fixed << std::setprecision(4) << "chi after 2 steps "
            << saved << ", restored " << restored << std::endl;
    deallog << "chi at step 3: uninterrupted " << reference
            << ", restarted with chi " << continued
            << ", restarted without chi " << from_zero << std::endl;
    deallog << "attached data sets: " << n_attached("with_chi") << " with chi, "
            << n_attached("without_chi") << " without, "
            << n_attached("after_discard") << " after dropping chi"
            << std::endl;
  }
}
