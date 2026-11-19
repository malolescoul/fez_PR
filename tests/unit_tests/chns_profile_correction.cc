#include <cahn_hilliard.h>

#include "../tests.h"

// Check the spatial divergence independently by differentiating the flux
// along a quadratic phase field, including near and beyond |phi|=1. The second
// phi-derivative jumps at |phi|=1, so centred differences must not straddle it
// (continuity there is checked in check_tails()).
template <int dim>
void check_flux()
{
  constexpr double epsilon = 0.3;
  Tensor<1, dim>   g;
  Tensor<2, dim>   H;
  for (unsigned int i = 0; i < dim; ++i)
  {
    g[i] = 0.2 + 0.1 * i;
    for (unsigned int j = 0; j < dim; ++j)
      H[i][j] = 0.03 * (i + j + 1);
  }
  for (const double phi : {0., -0.4, 0.9996, -0.9996, 0.9999, -1.01, 1.01})
  {
    const auto correction =
      CahnHilliard::evaluate_profile_correction(phi, g, H, epsilon);
    double       divergence = 0.;
    const double h          = 1e-8;
    for (unsigned int d = 0; d < dim; ++d)
    {
      Tensor<1, dim> dx;
      dx[d]           = h;
      const auto plus = CahnHilliard::evaluate_profile_correction(
        phi + g * dx + 0.5 * (dx * (H * dx)), g + H * dx, H, epsilon);
      const auto minus = CahnHilliard::evaluate_profile_correction(
        phi - g * dx + 0.5 * (dx * (H * dx)), g - H * dx, H, epsilon);
      divergence += (plus.flux[d] - minus.flux[d]) / (2. * h);
    }
    AssertThrow(std::isfinite(correction.divergence) &&
                  std::abs(divergence - correction.divergence) <
                    2e-6 * std::max(1., std::abs(divergence)),
                ExcMessage(
                  "Profile flux divergence differs from spatial differences."));
  }
  Tensor<1, dim> normal = g / g.norm();
  for (const double phi : {0., 0.4, 0.99, 0.999, 0.9996, 0.9999, 0.99999})
  {
    const double q      = (1. - phi * phi) / (std::sqrt(2.) * epsilon);
    const double dq     = -2. * phi / (std::sqrt(2.) * epsilon);
    const auto   result = CahnHilliard::evaluate_profile_correction(
      phi, q * normal, dq * q * outer_product(normal, normal), epsilon);
    AssertThrow(
      result.flux.norm() < 1e-13 && std::abs(result.divergence) < 1e-12,
      ExcMessage("The equilibrium tanh profile must not be corrected."));
  }
  const auto flat = CahnHilliard::evaluate_profile_correction(0.,
                                                              Tensor<1, dim>(),
                                                              Tensor<2, dim>(),
                                                              epsilon);
  AssertThrow(flat.flux.norm() == 0. && flat.divergence == 0.,
              ExcMessage("Zero gradient must give a finite zero flux."));
}

// Check every input of the analytical linearization, including the second
// derivatives needed by tracer SUPG near and beyond |phi|=1.
template <int dim>
void check_linearization()
{
  Tensor<1, dim> gradient;
  Tensor<2, dim> hessian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    gradient[d] = 0.2 + 0.1 * d;
    for (unsigned int e = 0; e < dim; ++e)
      hessian[d][e] = 0.03 * (d + e + 1);
  }
  for (const double phi :
       {0., -0.4, 0.99, -0.9996, 0.9996, -0.9999, 0.9999, 1.01, -1.01})
    for (const bool flat : {false, true})
    {
      // The flux is C1 at |phi|=1, but its second phi-derivative (divergence
      // linearization) jumps there: centred differences must not straddle it.
      const auto g = flat ? Tensor<1, dim>() : gradient;
      CahnHilliard::ProfileCorrectionLinearization<dim> jacobian;
      CahnHilliard::evaluate_profile_correction(
        phi, g, hessian, 0.3, true, &jacobian);
      for (unsigned int direction = 0; direction < 1 + dim + dim * dim;
           ++direction)
      {
        const double   dphi = direction == 0 ? 1. : 0.;
        Tensor<1, dim> dg;
        Tensor<2, dim> dh;
        if (direction > 0 && direction <= dim)
          dg[direction - 1] = 1.;
        if (direction > dim)
        {
          const unsigned int i = (direction - dim - 1) / dim;
          const unsigned int j = (direction - dim - 1) % dim;
          dh[i][j] += 0.5;
          dh[j][i] += 0.5;
        }
        const auto analytic = jacobian.variation(dphi, dg, dh);
        double     error    = std::numeric_limits<double>::max();
        for (const double step : {1e-6, 1e-7, 1e-8, 1e-9, 1e-10})
        {
          const auto plus = CahnHilliard::evaluate_profile_correction(
            phi + step * dphi, g + step * dg, hessian + step * dh, 0.3);
          const auto minus = CahnHilliard::evaluate_profile_correction(
            phi - step * dphi, g - step * dg, hessian - step * dh, 0.3);
          error = std::min(
            error,
            std::max(
              ((plus.flux - minus.flux) / (2. * step) - analytic.flux).norm() /
                std::max(1., analytic.flux.norm()),
              std::abs((plus.divergence - minus.divergence) / (2. * step) -
                       analytic.divergence) /
                std::max(1., std::abs(analytic.divergence))));
        }
        AssertThrow(error < 2e-6,
                    ExcMessage("Incorrect implicit profile linearization."));
      }
    }
}

// Tanh profile phi = tanh(s) for an affine inverse s with gradient w.
template <int dim>
struct TanhProfile
{
  double         phi;
  Tensor<1, dim> gradient;
  Tensor<2, dim> hessian;
  TanhProfile(const double s, const Tensor<1, dim> &w)
    : phi(std::tanh(s))
    , gradient((1. - phi * phi) * w)
    , hessian(-2. * phi * (1. - phi * phi) * outer_product(w, w))
  {}
};

// Check the tails: the gradient floor is negligible inside the interface, the
// tail flux stays bounded, equilibrium tails are not corrected, the flux is
// continuous across |phi|=1 and uniform pure phases are not anti-diffusive.
template <int dim>
void check_tails()
{
  constexpr double epsilon = 0.3;
  const double delta = CahnHilliard::profile_correction_gradient_floor(epsilon);
  Tensor<1, dim> w;
  w[0] = 0.7 / (std::sqrt(2.) * epsilon);
  const auto center =
    CahnHilliard::evaluate_profile_correction<dim>(0., w, {}, epsilon);
  // At fixed inverse slope, F/(1-phi^2) stays that of the centre.
  for (const double phi : {-.9, -.5, .5, .9})
  {
    const TanhProfile<dim> p(std::atanh(phi), w);
    const auto result = CahnHilliard::evaluate_profile_correction(p.phi,
                                                                  p.gradient,
                                                                  p.hessian,
                                                                  epsilon);
    AssertThrow((result.flux / (1. - phi * phi) - center.flux).norm() <
                  1e-3 * center.flux.norm(),
                ExcMessage("The correction is attenuated in the interface"));
  }
  for (const double phi : {-.99999999, -.9999, -.9996, .9996, .9999, .99999999})
  {
    const double           s = std::atanh(phi);
    const TanhProfile<dim> p(s, w);
    const auto result = CahnHilliard::evaluate_profile_correction(p.phi,
                                                                  p.gradient,
                                                                  p.hessian,
                                                                  epsilon);
    // Never stronger than the unfloored flux, and of the same sign.
    AssertThrow(result.flux * center.flux >= 0. &&
                  result.flux.norm() <=
                    (1. - phi * phi) * center.flux.norm() * (1. + 1e-12),
                ExcMessage("The floored tail flux must stay bounded"));
    // Independent spatial derivative along the normal.
    const double           h = 1e-5;
    const TanhProfile<dim> plus(s + h * w[0], w), minus(s - h * w[0], w);
    const auto             fp = CahnHilliard::evaluate_profile_correction(
      plus.phi, plus.gradient, plus.hessian, epsilon, false);
    const auto fm = CahnHilliard::evaluate_profile_correction(
      minus.phi, minus.gradient, minus.hessian, epsilon, false);
    AssertThrow(std::abs((fp.flux[0] - fm.flux[0]) / (2. * h) -
                         result.divergence) < 1e-10,
                ExcMessage("Incorrect tail divergence"));
    const TanhProfile<dim> equilibrium(s, w / .7);
    const auto             eq = CahnHilliard::evaluate_profile_correction(
      equilibrium.phi, equilibrium.gradient, equilibrium.hessian, epsilon);
    AssertThrow(eq.flux.norm() < 1e-13 && std::abs(eq.divergence) < 1e-12,
                ExcMessage("The equilibrium tail must not be corrected"));
  }
  // No branch at |phi|=1: a nonzero gradient gives a continuous flux.
  Tensor<1, dim> g;
  g[0] = 3. * delta;
  for (const double sign : {-1., 1.})
  {
    const auto inside = CahnHilliard::evaluate_profile_correction<dim>(
      sign * (1. - 1e-13), g, {}, epsilon);
    const auto outside = CahnHilliard::evaluate_profile_correction<dim>(
      sign * (1. + 1e-13), g, {}, epsilon);
    AssertThrow(inside.flux.norm() > 0. && (inside.flux - outside.flux).norm() <
                                             1e-10 * inside.flux.norm(),
                ExcMessage("The correction must be continuous at |phi|=1"));
  }
  // The anti-diffusive slope at zero gradient vanishes in pure phases,
  // whereas the unfloored slope would remain 1-sqrt(1+beta^2)/beta.
  for (const double phi : {-1., 1., -1.01, 1.01})
  {
    CahnHilliard::ProfileCorrectionLinearization<dim> jacobian;
    const auto result = CahnHilliard::evaluate_profile_correction<dim>(
      phi, {}, {}, epsilon, true, &jacobian);
    AssertThrow(result.flux.norm() == 0. && result.divergence == 0. &&
                  std::isfinite(jacobian.divergence_gradient.norm()) &&
                  jacobian.flux_gradient.norm() < 1e-2,
                ExcMessage("Uniform pure phases must not be anti-diffusive"));
  }
}

// The inverse profile used by the interface band is the exact atanh below its
// cutoff, and a C1 increasing continuation above.
void check_inverse_profile()
{
  const double cutoff = std::sqrt(1. - 0.25 * (1. - 0.999 * 0.999));
  for (const double phi :
       {0., 0.5, 0.99, 0.999, cutoff - 1e-6, cutoff + 1e-6, 1., 1.01})
    for (const double sign : {-1., 1.})
    {
      const auto   inverse = CahnHilliard::inverse_profile(sign * phi);
      const double step    = 1e-8;
      const double fd =
        (CahnHilliard::inverse_profile(sign * phi + step).value -
         CahnHilliard::inverse_profile(sign * phi - step).value) /
        (2. * step);
      AssertThrow(std::isfinite(inverse.value) && inverse.derivative > 0. &&
                    std::abs(fd - inverse.derivative) <
                      2e-6 * inverse.derivative,
                  ExcMessage("Incorrect inverse profile derivative."));
      if (phi <= cutoff)
        AssertThrow(std::abs(std::tanh(inverse.value) - sign * phi) < 1e-14,
                    ExcMessage("The inverse profile must be exact."));
    }
}

int main()
{
  initlog();
  check_flux<2>();
  check_flux<3>();
  check_linearization<2>();
  check_linearization<3>();
  check_tails<2>();
  check_tails<3>();
  check_inverse_profile();
  deallog << "Profile flux: divergence, equilibrium, tails and zero gradient OK"
          << std::endl;
}
