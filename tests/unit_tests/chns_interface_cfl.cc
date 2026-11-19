#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_tools.h>

#include "../tests.h"

/**
 * Check the CHNS interface CFL number dt * max |(u - w) . n| / epsilon on a
 * planar tanh profile normal to x, with components (u, x, phi).
 * Only the normal velocity relative to the mesh counts, and only where
 * |phi| < 0.99.
 */
constexpr double epsilon = 0.1;
constexpr double dt      = 0.01;

// Velocity, position = X + shift, and the planar profile.
class Fields : public Function<2>
{
public:
  Fields(const Tensor<1, 2> &velocity,
         const double        velocity_slope,
         const Tensor<1, 2> &shift)
    : Function<2>(5)
    , velocity(velocity)
    , velocity_slope(velocity_slope)
    , shift(shift)
  {}

  double value(const Point<2> &p, const unsigned int c) const override
  {
    if (c < 2)
      return velocity[c] + (c == 0 ? velocity_slope * p[0] : 0.);
    if (c < 4)
      return p[c - 2] + shift[c - 2];
    return std::tanh(p[0] / (std::sqrt(2.) * epsilon));
  }

private:
  Tensor<1, 2> velocity;
  double       velocity_slope;
  Tensor<1, 2> shift;
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();

  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation, -1., 1.);
  triangulation.refine_global(5);
  const FESystem<2> fe(FE_Q<2>(2) ^ 2, FE_Q<2>(1) ^ 2, FE_Q<2>(2));
  DoFHandler<2>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const MappingQ<2> mapping(1);
  const QGauss<2>   quadrature(4);

  const FEValuesExtractors::Vector velocity(0), position(2);
  const FEValuesExtractors::Scalar tracer(4);

  // Returns the interface CFL number, with the mesh velocity w = shift / dt
  // from BDF1 if with_mesh_velocity, and w = 0 otherwise.
  const auto cfl = [&](const Tensor<1, 2> &u,
                       const double        slope,
                       const Tensor<1, 2> &shift,
                       const bool          with_mesh_velocity) {
    Vector<double> present(dof_handler.n_dofs()), previous(present.size());
    VectorTools::interpolate(mapping,
                             dof_handler,
                             Fields(u, slope, shift),
                             present);
    VectorTools::interpolate(mapping,
                             dof_handler,
                             Fields(u, slope, Tensor<1, 2>()),
                             previous);
    return PostProcessingTools::compute_max_interface_cfl(
      dt,
      mapping,
      dof_handler,
      quadrature,
      present,
      velocity,
      tracer,
      epsilon,
      with_mesh_velocity ? &position : nullptr,
      std::vector<double>{1. / dt, -1. / dt},
      std::vector<Vector<double>>{previous});
  };

  const auto check = [](const bool condition, const std::string &message) {
    AssertThrow(condition, ExcMessage(message));
  };

  const Tensor<1, 2> u({2., 1.}), tangential({0., 3.});

  // Normal velocity 2: dt * 2 / epsilon = 0.2.
  check(std::abs(cfl(u, 0., Tensor<1, 2>(), false) - 0.2) < 1e-12,
        "Normal velocity");
  // Velocity tangent to the interface does not move the profile.
  check(cfl(tangential, 0., Tensor<1, 2>(), false) < 1e-12,
        "Tangential velocity");
  // A mesh moving with the normal velocity cancels it, and a mesh moving
  // against it adds to it: |2 - (-1)| * dt / epsilon = 0.3.
  check(cfl(u, 0., Tensor<1, 2>({2. * dt, 0.}), true) < 1e-12,
        "Mesh velocity equal to the normal velocity");
  check(std::abs(cfl(u, 0., Tensor<1, 2>({-dt, 0.}), true) - 0.3) < 1e-12,
        "Mesh velocity opposite to the normal velocity");
  // u_x = 4x: only |phi| < 0.99, i.e. |x| < sqrt(2) eps atanh(0.99), counts,
  // whereas the velocity reaches 4 at the boundary.
  const double edge     = std::sqrt(2.) * epsilon * std::atanh(0.99);
  const double expected = dt * 4. * edge / epsilon;
  const double banded   = cfl(Tensor<1, 2>(), 4., Tensor<1, 2>(), false);
  check(banded > 0.9 * expected && banded < 1.02 * expected,
        "Restriction to |phi| < 0.99");

  deallog << "Interface CFL: normal, tangential, mesh velocity and band OK"
          << std::endl;
}
