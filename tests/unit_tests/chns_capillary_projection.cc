#include <chns_capillary_projection.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

class Phase : public Function<2>
{
public:
  bool   curved = false;
  double value(const Point<2> &p, const unsigned int = 0) const override
  {
    return curved ? (p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5) :
                    p[0] + 2 * p[1];
  }
};
void check(const bool adaptive)
{
  parallel::shared::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);
  if (adaptive)
  {
    tria.begin_active()->set_refine_flag();
    tria.execute_coarsening_and_refinement();
  }
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<2>      mapping;
  QGauss<2>         quad(3);
  LA::ParVectorType owned, sol;
  owned.reinit(dofs.locally_owned_dofs(), MPI_COMM_WORLD);
  sol.reinit(dofs.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dofs),
             MPI_COMM_WORLD);
  VectorTools::interpolate(mapping, dofs, Phase(), owned);
  sol = owned;
  CHNSSurfaceTension::WeakForceProjection<2> projection(tria);
  const auto result = projection.compute(dofs, mapping, quad, sol, 0, 0.072);
  AssertThrow(result.first.norm() < 1e-12 && result.second < 1e-12,
              ExcMessage(
                "Planar force must cancel including boundary traction"));
  Phase curved;
  curved.curved = true;
  VectorTools::interpolate(mapping, dofs, curved, owned);
  sol = owned;
  AffineConstraints<double> constraints(
    dofs.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dofs));
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  constraints.close();
  constraints.distribute(owned);
  sol = owned;
  const auto curved_result =
    projection.compute(dofs, mapping, quad, sol, 0, 0.072);
  AssertThrow(
    (adaptive || curved_result.first.norm() < 1e-12) &&
      curved_result.second > 0.01,
    ExcMessage("P1 weak force must retain stress jumps on a curved interface"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << (adaptive ? "Adaptive" : "Uniform")
            << " weak force including boundaries OK" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  check(false);
  check(true);
}
