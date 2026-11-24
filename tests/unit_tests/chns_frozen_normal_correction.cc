#include <cahn_hilliard.h>

#include "../tests.h"

// Profile flux with the normal frozen at the previous time step:
// F = w (grad(phi).e - q(phi)) e. The current state and the previous state
// are two different quadratic phase fields, so that the frozen normal varies
// in space independently of phi.

constexpr double epsilon = 0.3;

template <int dim>
struct State
{
  double         phi;
  Tensor<1, dim> g;
  Tensor<2, dim> H;
  Tensor<1, dim> g_previous;
  Tensor<2, dim> H_previous;

  // Both quadratic fields shifted by dx.
  State shifted(const Tensor<1, dim> &dx) const
  {
    return {phi + g * dx + 0.5 * (dx * (H * dx)),
            g + H * dx,
            H,
            g_previous + H_previous * dx,
            H_previous};
  }

  CahnHilliard::ProfileCorrection<dim>
  evaluate(CahnHilliard::ProfileCorrectionLinearization<dim> *linearization =
             nullptr) const
  {
    return CahnHilliard::evaluate_frozen_normal_correction(
      phi,
      g,
      H,
      epsilon,
      CahnHilliard::frozen_normal(g_previous, H_previous, epsilon),
      true,
      linearization);
  }
};

template <int dim>
State<dim> make_state(const double phi)
{
  State<dim> s;
  s.phi = phi;
  for (unsigned int i = 0; i < dim; ++i)
  {
    s.g[i]          = 0.2 + 0.1 * i;
    s.g_previous[i] = 0.3 - 0.05 * i;
    for (unsigned int j = 0; j < dim; ++j)
    {
      s.H[i][j]          = 0.03 * (i + j + 1);
      s.H_previous[i][j] = 0.05 * (i + 1) * (j + 1) - 0.02;
    }
  }
  return s;
}

// The divergence, including the gradients of the frozen normal and of its
// weight, against centred differences of the flux in space. q is smooth
// beyond |phi| = 1, so these points are included.
template <int dim>
void check_divergence()
{
  double error = 0.;
  for (const double phi : {0., -0.4, 0.9996, -0.9999, 1.01, -1.01})
  {
    const auto   s          = make_state<dim>(phi);
    const auto   correction = s.evaluate();
    double       divergence = 0.;
    const double h          = 1e-6;
    for (unsigned int d = 0; d < dim; ++d)
    {
      Tensor<1, dim> dx;
      dx[d] = h;
      divergence +=
        (s.shifted(dx).evaluate().flux[d] - s.shifted(-dx).evaluate().flux[d]) /
        (2. * h);
    }
    error = std::max(error,
                     std::abs(divergence - correction.divergence) /
                       std::max(1., std::abs(divergence)));
  }
  AssertThrow(error < 1e-7,
              ExcMessage("Frozen-normal flux divergence differs from spatial "
                         "differences."));
  deallog << dim << "D divergence against spatial differences: OK" << std::endl;
}

// Derivatives with respect to phi, grad(phi) and the Hessian of phi, the
// normal being fixed.
template <int dim>
void check_linearization()
{
  double error = 0.;
  for (const double phi : {0., 0.5, -0.9996, 1.01})
  {
    const auto                                        s = make_state<dim>(phi);
    CahnHilliard::ProfileCorrectionLinearization<dim> linearization;
    s.evaluate(&linearization);
    const double   dphi = 0.7;
    Tensor<1, dim> dg;
    Tensor<2, dim> dH;
    for (unsigned int i = 0; i < dim; ++i)
    {
      dg[i] = 0.3 - 0.2 * i;
      for (unsigned int j = 0; j < dim; ++j)
        dH[i][j] = 0.1 * (i + 1) - 0.05 * j;
    }
    const auto   variation = linearization.variation(dphi, dg, dH);
    const double h         = 1e-6;
    auto         plus = s, minus = s;
    plus.phi += h * dphi;
    plus.g += h * dg;
    plus.H += h * dH;
    minus.phi -= h * dphi;
    minus.g -= h * dg;
    minus.H -= h * dH;
    const auto fp = plus.evaluate(), fm = minus.evaluate();
    error = std::max({error,
                      ((fp.flux - fm.flux) / (2. * h) - variation.flux).norm(),
                      std::abs((fp.divergence - fm.divergence) / (2. * h) -
                               variation.divergence)});
  }
  AssertThrow(error < 1e-7,
              ExcMessage("Frozen-normal linearization differs from finite "
                         "differences."));
  deallog << dim << "D linearization against finite differences: OK"
          << std::endl;
}

// The tanh profile, with the normal of the same state, is not corrected; a
// slope q (1 + eta) along it gives the flux gain * eta * q.
template <int dim>
void check_equilibrium()
{
  Tensor<1, dim> n;
  n[0]             = 0.6;
  n[dim - 1]       = 0.8;
  double max_error = 0.;
  for (const double phi : {0., 0.4, 0.99, 0.9999, 0.99999})
  {
    const double c = 1. / (std::sqrt(2.) * epsilon);
    const double q = c * (1. - phi * phi), dq = -2. * c * phi;
    const auto   g      = q * n;
    const auto   H      = Tensor<2, dim>(dq * q * outer_product(n, n));
    const auto   normal = CahnHilliard::frozen_normal(g, H, epsilon);
    const auto   result = CahnHilliard::evaluate_frozen_normal_correction(
      phi, g, H, epsilon, normal);
    AssertThrow(result.flux.norm() < 1e-13 &&
                  std::abs(result.divergence) < 1e-12,
                ExcMessage("The equilibrium tanh profile must not be "
                           "corrected."));
    const double eta     = 0.1;
    const auto   steeper = CahnHilliard::evaluate_frozen_normal_correction(
      phi, (1. + eta) * g, H, epsilon, normal, false);
    const double gain =
      CahnHilliard::profile_correction_gain(phi, epsilon, true);
    max_error =
      std::max(max_error, std::abs(steeper.flux * n - gain * eta * q));
  }
  AssertThrow(max_error < 1e-12,
              ExcMessage("Linear gain of the frozen-normal flux."));
  const auto flat = CahnHilliard::evaluate_frozen_normal_correction(
    0.5,
    Tensor<1, dim>(),
    Tensor<2, dim>(),
    epsilon,
    CahnHilliard::frozen_normal(Tensor<1, dim>(), Tensor<2, dim>(), epsilon));
  AssertThrow(flat.flux.norm() == 0. && flat.divergence == 0.,
              ExcMessage("A vanishing previous gradient switches the flux "
                         "off."));
  deallog << dim << "D equilibrium, gain and flat phase: OK" << std::endl;
}

// Diffusion tensor dF/d(grad phi) at a filament core: phi = 0.88 with a slope
// far below q. With the implicit normal its tangential eigenvalue is
// 1 - Q/R < 0 (anti-diffusion); with the frozen normal it is zero.
void check_filament_core()
{
  constexpr int  dim = 2;
  const double   phi = 0.88;
  const double   q   = (1. - phi * phi) / (std::sqrt(2.) * epsilon);
  Tensor<1, dim> g;
  g[1] = 0.01 * q;
  CahnHilliard::ProfileCorrectionLinearization<dim> implicit, frozen;
  CahnHilliard::evaluate_profile_correction<dim>(
    phi, g, Tensor<2, dim>(), epsilon, false, &implicit);
  CahnHilliard::evaluate_frozen_normal_correction<dim>(
    phi,
    g,
    Tensor<2, dim>(),
    epsilon,
    CahnHilliard::frozen_normal(g, Tensor<2, dim>(), epsilon),
    false,
    &frozen);
  deallog << std::fixed << std::setprecision(3)
          << "filament core, phi = 0.88, |grad phi| = 0.01 q: tangential / "
             "normal diffusion, implicit normal "
          << implicit.flux_gradient[0][0] << " / "
          << implicit.flux_gradient[1][1] << ", frozen normal "
          << frozen.flux_gradient[0][0] << " / " << frozen.flux_gradient[1][1]
          << std::endl;
}

int main()
{
  initlog();
  check_divergence<2>();
  check_divergence<3>();
  check_linearization<2>();
  check_linearization<3>();
  check_equilibrium<2>();
  check_equilibrium<3>();
  check_filament_core();
}
