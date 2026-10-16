#include <cahn_hilliard.h>

#include "../tests.h"

// Check the spatial divergence independently by differentiating the flux
// along a quadratic phase field, including the C2 activation interval.
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
  for (const double phi : {0., -0.4, 0.9996, -0.9996, 1., -1., 1.01})
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
    if (std::abs(phi) >= 1.)
      AssertThrow(correction.flux.norm() == 0. && correction.divergence == 0.,
                  ExcMessage("Correction must vanish outside the interface."));
  }
  Tensor<1, dim> normal = g / g.norm();
  for (const double phi : {0., 0.4, 0.99, 0.999, 0.9996})
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
// derivatives needed by tracer SUPG inside the narrow activation interval.
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
  for (const double phi : {0., -0.4, 0.99, -0.9996, 0.9996, 1., 1.01})
    for (const bool flat : {false, true})
    {
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
        for (const double step : {1e-6, 1e-7, 1e-8})
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

void check_inverse_profile()
{
  const double cutoff =
    std::sqrt(1. - 0.25 * CahnHilliard::profile_correction_tail);
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
                  ExcMessage("Incorrect regularized inverse derivative."));
      if (phi <= cutoff)
        AssertThrow(std::abs(std::tanh(inverse.value) - sign * phi) < 1e-14,
                    ExcMessage(
                      "The retained profile must use the exact inverse."));
    }
}

template <int dim>
void check_reconstruction()
{
  // A nodal equilibrium tanh reconstructs the affine inverse exactly, even
  // though the polynomial interpolant of the phase is not an equilibrium.
  Tensor<1, dim> normal;
  normal[0]                = 1.;
  constexpr double epsilon = 0.3;
  const double     left = std::tanh(-2.), right = std::tanh(2.);
  for (const double weight : {0.1, 0.3, 0.6, 0.9})
  {
    const double inverse =
      (1. - weight) * CahnHilliard::inverse_profile(left).value +
      weight * CahnHilliard::inverse_profile(right).value;
    CahnHilliard::ReconstructedProfile<dim> profile{
      std::tanh(inverse), normal / (std::sqrt(2.) * epsilon), {}};
    const auto result = CahnHilliard::evaluate_profile_correction(
      profile.value, profile.gradient(), profile.hessian(), epsilon);
    AssertThrow(
      result.flux.norm() < 1e-13 && std::abs(result.divergence) < 1e-12,
      ExcMessage(
        "Nodal planar equilibrium must have zero reconstructed correction."));
  }
  Tensor<1, dim> g, dg;
  Tensor<2, dim> H, dH;
  for (unsigned int d = 0; d < dim; ++d)
  {
    g[d]  = 0.2 + 0.1 * d;
    dg[d] = 0.3 - 0.1 * d;
    for (unsigned int e = 0; e < dim; ++e)
    {
      H[d][e]  = 0.03 * (d + e + 1);
      dH[d][e] = 0.02 * (d + e + 1);
    }
  }
  for (const double inverse : {-5., -0.4, 0., 0.5, 5.})
  {
    const CahnHilliard::ReconstructedProfile<dim> profile{std::tanh(inverse),
                                                          g,
                                                          H};
    const auto   analytic = profile.variation(0.3, dg, dH, true);
    const double h        = 1e-6;
    const CahnHilliard::ReconstructedProfile<dim> plus{
      std::tanh(inverse + 0.3 * h), g + h * dg, H + h * dH};
    const CahnHilliard::ReconstructedProfile<dim> minus{
      std::tanh(inverse - 0.3 * h), g - h * dg, H - h * dH};
    AssertThrow(
      std::abs((plus.value - minus.value) / (2. * h) - analytic.value) < 1e-9 &&
        ((plus.gradient() - minus.gradient()) / (2. * h) - analytic.gradient)
            .norm() < 1e-9 &&
        ((plus.hessian() - minus.hessian()) / (2. * h) - analytic.hessian)
            .norm() < 1e-9,
      ExcMessage("Incorrect reconstructed phase variation."));
  }
}

int main()
{
  initlog();
  check_inverse_profile();
  check_reconstruction<2>();
  check_reconstruction<3>();
  check_flux<2>();
  check_flux<3>();
  check_linearization<2>();
  check_linearization<3>();
  deallog << "Profile flux: divergence, equilibrium, tails and zero gradient OK"
          << std::endl;
}
