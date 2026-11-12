#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <stabilization_tools.h>

#include "../../tests.h"

using namespace dealii;

void check(const bool condition, const char *message)
{
  AssertThrow(condition, ExcMessage(message));
}

void test_continuity(const TimeHandler &time)
{
  for (const unsigned int degree : {1, 2, 3})
  {
    const FE_SimplexP<2>   fe(degree);
    const QGaussSimplex<2> quadrature(degree + 1);
    for (const double aspect : {1., 10., 100.})
    {
      Tensor<2, 2> inverse_jacobian;
      inverse_jacobian[0][0] = 8.;
      inverse_jacobian[1][1] = 8. * aspect;
      for (const auto &point : quadrature.get_points())
      {
        std::vector<Tensor<1, 2>> gradients(fe.n_dofs_per_cell());
        for (unsigned int i = 0; i < gradients.size(); ++i)
          gradients[i] = transpose(inverse_jacobian) * fe.shape_grad(i, point);
        const auto tau = [&](const Tensor<1, 2> &velocity) {
          return StabilizationTools::compute_tau_supg_2d(
            time, degree, .1, velocity, gradients, inverse_jacobian, true);
        };
        Tensor<1, 2> below, above;
        below[1] = .999999e-6;
        above[1] = 1.000001e-6;
        check(std::abs(tau(above) / tau(below) - 1.) < 1e-8,
              "Tau jumps at the former velocity cutoff.");

        const double at_rest = tau(Tensor<1, 2>());
        for (const Tensor<1, 2> velocity : {Tensor<1, 2>({1e-10, 0.}),
                                            Tensor<1, 2>({0., 1e-10}),
                                            Tensor<1, 2>({-1e-10, 1e-10})})
          check(std::abs(tau(velocity) / at_rest - 1.) < 1e-12,
                "Tau has a direction-dependent limit at rest.");
      }
    }
  }
}

void test_geometry_and_convection(const TimeHandler &steady)
{
  Tensor<2, 2> inverse_jacobian;
  inverse_jacobian[0][0]              = 1.;
  inverse_jacobian[1][1]              = 2.;
  std::vector<Tensor<1, 2>> gradients = {Tensor<1, 2>({-1., -2.}),
                                         Tensor<1, 2>({1., 0.}),
                                         Tensor<1, 2>({0., 2.})};
  const auto                tau       = [&](const Tensor<1, 2> &velocity,
                       const bool          simplex,
                       const double        viscosity) {
    return StabilizationTools::compute_tau_supg_2d(
      steady, 1, viscosity, velocity, gradients, inverse_jacobian, simplex);
  };

  // Right triangle with legs 1 and 1/2: its smallest height is 1/sqrt(5).
  // The corresponding rectangle has smallest height 1/2.
  const double triangle  = tau(Tensor<1, 2>(), true, 1.);
  const double rectangle = tau(Tensor<1, 2>(), false, 1.);
  check(std::abs(triangle - 1. / 60.) < 1e-14, "Wrong triangle height.");
  check(std::abs(rectangle - 1. / 48.) < 1e-14, "Wrong rectangle height.");

  const Tensor<1, 2> velocity({.2, -.3});
  const double       old_convective = StabilizationTools::compute_tau_supg(
    steady, gradients.size(), std::sqrt(1.25), 1, 0., velocity, gradients);
  check(std::abs(tau(velocity, true, 0.) / old_convective - 1.) < 1e-14,
        "Separating diffusion changed the convective frequency.");

  // Rotate the cell and velocity together. Tau must not depend on the axes.
  Tensor<2, 2> rotation;
  rotation[0][0] = rotation[1][1] = std::cos(.37);
  rotation[1][0]                  = std::sin(.37);
  rotation[0][1]                  = -rotation[1][0];
  const double unrotated          = tau(velocity, true, .1);
  inverse_jacobian                = inverse_jacobian * transpose(rotation);
  for (auto &gradient : gradients)
    gradient = rotation * gradient;
  check(std::abs(tau(rotation * velocity, true, .1) / unrotated - 1.) < 1e-14,
        "Tau depends on the physical coordinate axes.");

  // Doubling the geometry multiplies the viscous time scale at rest by four.
  inverse_jacobian *= .5;
  check(std::abs(tau(Tensor<1, 2>(), true, 1.) / triangle - 4.) < 1e-13,
        "The diffusive length does not follow the physical mapping.");
}

int main()
{
  initlog();
  Parameters::TimeIntegration parameters{};
  parameters.scheme           = Parameters::TimeIntegration::Scheme::BDF1;
  parameters.dt               = .001;
  parameters.t_initial        = 0.;
  parameters.t_end            = 1.;
  parameters.n_time_intervals = 1;
  const TimeHandler transient(parameters);
  parameters.scheme = Parameters::TimeIntegration::Scheme::stationary;
  const TimeHandler steady(parameters);
  test_continuity(transient);
  test_continuity(steady);
  test_geometry_and_convection(steady);
  deallog << "Continuous tau at rest and across the former cutoff: OK"
          << std::endl;
  deallog << "Geometry, rotation, scaling and convection: OK" << std::endl;
}
