#include <chns_profile_correction_cutoff.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

// Strain scaling of the profile correction on a mesh. A planar interface of
// normal n is placed in a linear flow: the normal strain n.grad(u).n is
// measured over the interface band, and its positive part sets the ratio
// r = clamp(safety * strain / reference, 1, max ratio).

constexpr double epsilon = 1.;

enum class Flow
{
  stretching,  // u = a ((n.x) n - (t.x) t): the isolines move apart
  compression, // opposite flow: the isolines move closer
  shear        // u = a (n.x) t: no normal strain
};

// Velocity (components 0 and 1) and phase tanh(n.x / (sqrt(2) eps))
// (component 2).
class State : public Function<2>
{
public:
  State(const double angle, const Flow flow, const double a)
    : Function<2>(3)
    , n({std::sin(angle), std::cos(angle)})
    , t({std::cos(angle), -std::sin(angle)})
    , flow(flow)
    , a(a)
  {}

  double value(const Point<2> &p, const unsigned int component) const override
  {
    const double xn = n * p, xt = t * p;
    if (component == 2)
      return std::tanh(xn / (std::sqrt(2.) * epsilon));
    Tensor<1, 2> u;
    switch (flow)
    {
      case Flow::stretching:
        u = a * (xn * n - xt * t);
        break;
      case Flow::compression:
        u = -a * (xn * n - xt * t);
        break;
      default:
        u = a * xn * t;
    }
    return u[component];
  }

private:
  Tensor<1, 2> n, t;
  Flow         flow;
  double       a;
};

void check(const double       angle,
           const Flow         flow,
           const double       a,
           const std::string &name)
{
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {16, 16},
                                            Point<2>(-6., -6.),
                                            Point<2>(6., 6.));
  const FESystem<2> fe(FE_Q<2>(2) ^ 2, FE_Q<2>(2));
  DoFHandler<2>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  const MappingQ<2> mapping(1);
  const QGauss<2>   quadrature(3);

  const IndexSet owned = dof_handler.locally_owned_dofs();
  const IndexSet relevant =
    DoFTools::extract_locally_relevant_dofs(dof_handler);
  LA::ParVectorType interpolated(owned, MPI_COMM_WORLD);
  VectorTools::interpolate(mapping,
                           dof_handler,
                           State(angle, flow, a),
                           interpolated);
  LA::ParVectorType solution(owned, relevant, MPI_COMM_WORLD);
  solution = interpolated;

  CahnHilliard::InteractionCutoffParameters parameters;
  parameters.cutoff           = false;
  parameters.strain_scaling   = true;
  parameters.strain_reference = 1.;
  parameters.strain_safety    = 2.;
  parameters.max_strain_ratio = 10.;
  CahnHilliard::ProfileCorrectionCutoff<2> cutoff(tria, parameters);
  const auto                               stats = cutoff.update(
    dof_handler, mapping, quadrature, solution, 2, epsilon, 1, 1e-3, 0);

  // Range of the weight r (1 - chi) = r near the interface, away from the
  // boundary.
  const Tensor<1, 2> t({std::cos(angle), -std::sin(angle)});
  const Tensor<1, 2> n({std::sin(angle), std::cos(angle)});
  FEValues<2> fe_values(mapping, fe, quadrature, update_quadrature_points);
  double      low = 1e10, high = 0.;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        const auto &p = fe_values.quadrature_point(q);
        if (std::abs(n * p) > 2. * epsilon || std::abs(t * p) > 3. * epsilon)
          continue;
        const double w = cutoff.weight(cell->active_cell_index(), q);
        low            = std::min(low, w);
        high           = std::max(high, w);
      }
    }
  low  = Utilities::MPI::min(low, MPI_COMM_WORLD);
  high = Utilities::MPI::max(high, MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << name << ": max strain " << std::fixed << std::setprecision(3)
            << stats.max_strain << ", ratio near the interface in [" << low
            << ", " << high << "]" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();

  // Stretching below, at and above the saturation of the ratio.
  for (const double a : {1., 3., 10.})
    check(0., Flow::stretching, a, "stretching a = " + std::to_string(int(a)));
  // Interface inclined on the mesh: the normal comes from the recovered
  // gradient of psi.
  check(numbers::PI / 6., Flow::stretching, 3., "inclined stretching a = 3");
  // Neither compression nor shear spreads the isolines.
  check(0., Flow::compression, 3., "compression a = 3");
  check(numbers::PI / 6., Flow::shear, 3., "inclined shear a = 3");
}
