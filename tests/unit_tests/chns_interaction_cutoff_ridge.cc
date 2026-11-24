#include <chns_profile_correction_cutoff.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

// Cutoff field on a mesh for filaments along x. In a filament, the profile
// correction keeps each side on the tanh of the distance to its own interface,
// with a kink of grad(phi) on the midline. With the midline on a face, as on
// the symmetry axis of a jet, this kink is invisible to the Hessian of phi
// inside the elements, and must be detected through the recovered gradient.

constexpr double epsilon = 1.;
constexpr double width   = 0.75 * epsilon; // cell size, about the jet's

enum class Shape
{
  planar,   // single interface at y = 0
  distance, // tanh of the distance to the nearest interface
  overlap   // sum of two tanh profiles
};

class FilamentFunction : public Function<2>
{
public:
  FilamentFunction(const Shape  shape,
                   const double a,
                   const double centre,
                   const double sign)
    : shape(shape)
    , a(a)
    , centre(centre)
    , sign(sign)
  {}

  double value(const Point<2> &p, const unsigned int = 0) const override
  {
    const double y = p[1] - centre;
    const double s = std::sqrt(2.) * epsilon;
    switch (shape)
    {
      case Shape::planar:
        return std::tanh(y / s);
      case Shape::distance:
        return sign * std::tanh((0.5 * a - std::abs(y)) / s);
      default:
        return sign * (std::tanh((y + 0.5 * a) / s) +
                       std::tanh((0.5 * a - y) / s) - 1.);
    }
  }

private:
  Shape  shape;
  double a, centre, sign;
};

void check(const Shape        shape,
           const double       a,
           const double       centre,
           const double       sign,
           const std::string &name)
{
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  // 16 cells of 0.75 eps across: y = 0 is a face.
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {16, 16},
                                            Point<2>(-6., -6.),
                                            Point<2>(6., 6.));
  const FE_Q<2> fe(2);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  const MappingQ<2> mapping(1);
  const QGauss<2>   quadrature(3);

  const IndexSet owned = dof_handler.locally_owned_dofs();
  const IndexSet relevant =
    DoFTools::extract_locally_relevant_dofs(dof_handler);
  LA::ParVectorType interpolated(owned, MPI_COMM_WORLD);
  VectorTools::interpolate(mapping,
                           dof_handler,
                           FilamentFunction(shape, a, centre, sign),
                           interpolated);
  LA::ParVectorType solution(owned, relevant, MPI_COMM_WORLD);
  solution = interpolated;

  CahnHilliard::ProfileCorrectionCutoff<2> cutoff(tria, {});
  cutoff.update(
    dof_handler, mapping, quadrature, solution, 0, epsilon, 1, 1e-3);

  // Largest chi inside the filament (|y| < a/2) and on the outer side, more
  // than one epsilon away from the interfaces.
  FEValues<2> fe_values(mapping, fe, quadrature, update_quadrature_points);
  double      inside = 0., outside = 0.;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        const double y   = std::abs(fe_values.quadrature_point(q)[1] - centre);
        const double chi = cutoff.chi(cell->active_cell_index(), q);
        if (y < 0.5 * a)
          inside = std::max(inside, chi);
        else if (y > 0.5 * a + epsilon)
          outside = std::max(outside, chi);
      }
    }
  inside  = Utilities::MPI::max(inside, MPI_COMM_WORLD);
  outside = Utilities::MPI::max(outside, MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << name << ": chi inside " << std::fixed << std::setprecision(2)
            << inside << ", outside " << outside << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();

  // A planar interface keeps the correction.
  check(Shape::planar, 0., 0., 1., "planar tanh");
  // Distance profiles: kept while the filament is wide, cut once thin,
  // whether the midline lies on a face or in the middle of the cells.
  check(Shape::distance, 7., 0., 1., "distance profile, a = 7 eps, on a face");
  for (const double centre : {0., 0.5 * width})
    for (const double a : {5., 4.})
      check(Shape::distance,
            a,
            centre,
            1.,
            "distance profile, a = " + std::to_string(int(a)) + " eps, " +
              (centre == 0. ? "on a face" : "inside cells"));
  // Gas filament: the side mask cuts the phase phi < 0.
  check(Shape::distance, 4., 0., -1., "gas distance profile, a = 4 eps");
  // Overlapping tanh profiles, as without the correction.
  check(Shape::overlap, 4., 0., 1., "two tanh, a = 4 eps, on a face");
}
