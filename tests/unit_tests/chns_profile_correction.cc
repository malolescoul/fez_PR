#include <cahn_hilliard.h>

#include "../tests.h"

// Check the spatial divergence independently by differentiating the flux
// along a quadratic phase field, including the activation interval.
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
  for (const double phi :
       {0., -0.4, 0.9996, -0.9996, 0.99986, -0.99986, 1., -1., 1.01})
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
// derivatives needed by tracer SUPG around the activation interval.
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
  for (const double phi : {0., -0.4, 0.99, -0.9996, 0.9996, 0.9999, 1.01})
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
  const double cutoff = CahnHilliard::profile_correction_cutoff;
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
      inverse, normal / (std::sqrt(2.) * epsilon), {}};
    const auto result =
      CahnHilliard::evaluate_profile_correction(profile, epsilon);
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
    const CahnHilliard::ReconstructedProfile<dim> profile{inverse, g, H};
    const auto   analytic = profile.variation(0.3, dg, dH, true);
    const double h        = 1e-6;
    const CahnHilliard::ReconstructedProfile<dim> plus{inverse + 0.3 * h,
                                                       g + h * dg,
                                                       H + h * dH};
    const CahnHilliard::ReconstructedProfile<dim> minus{inverse - 0.3 * h,
                                                        g - h * dg,
                                                        H - h * dH};
    AssertThrow(
      std::abs((plus.value - minus.value) / (2. * h) - analytic.value) < 1e-9 &&
        ((plus.gradient() - minus.gradient()) / (2. * h) - analytic.gradient)
            .norm() < 1e-9 &&
        ((plus.hessian() - minus.hessian()) / (2. * h) - analytic.hessian)
            .norm() < 1e-9,
      ExcMessage("Incorrect reconstructed phase variation."));
  }
}

// Spatial and input differences in inverse-profile coordinates remain
// meaningful in the transition and in the inactive pure phases. Scale
// tolerances by sech(s)^2 to resolve the active tails.
template <int dim>
void check_factorized_correction()
{
  Tensor<1, dim> G;
  Tensor<2, dim> H;
  for (unsigned int i = 0; i < dim; ++i)
  {
    G[i] = 0.2 + 0.1 * i;
    for (unsigned int j = 0; j < dim; ++j)
      H[i][j] = 0.03 * (i + j + 1);
  }
  for (const double epsilon : {0.004, 0.3})
    for (const double s :
         {-50., -20., -8., -4.8, -4.7, -0.4, 0., 0.5, 4.7, 4.8, 8., 20., 50.})
      for (const bool flat : {false, true})
      {
        const auto g = flat ? Tensor<1, dim>() : G;
        const CahnHilliard::ReconstructedProfile<dim>     profile{s, g, H};
        CahnHilliard::ProfileCorrectionLinearization<dim> jacobian;
        const auto result = CahnHilliard::evaluate_profile_correction(
          profile, epsilon, true, &jacobian);
        const double scale = profile.phase_factor;
        AssertThrow(std::isfinite(result.divergence) && scale > 0.,
                    ExcMessage("The reconstructed tail must remain finite."));
        if (std::abs(profile.value) >= CahnHilliard::profile_correction_cutoff)
          AssertThrow(result.flux.norm() == 0. && result.divergence == 0. &&
                        jacobian.flux_gradient.norm() == 0. &&
                        jacobian.flux_value.norm() == 0. &&
                        jacobian.divergence_value == 0. &&
                        jacobian.divergence_gradient.norm() == 0.,
                      ExcMessage("Inactive tails must have zero flux and Jacobian."));
        else if (!flat)
          AssertThrow(result.flux * g < 0.,
                      ExcMessage(
                        "An over-wide reconstructed tail needs correction."));

        // Compare with the original, direct-phase expression where it is
        // numerically well conditioned (no inverse-coordinate formula here).
        if (std::abs(s) < 4.)
        {
          const auto direct = CahnHilliard::evaluate_profile_correction(
            profile.value, profile.gradient(), profile.hessian(), epsilon);
          AssertThrow((direct.flux - result.flux).norm() < 1e-11 * scale &&
                        std::abs(direct.divergence - result.divergence) <
                          1e-10 * scale,
                      ExcMessage("Factorization changed the interior flux."));
        }

        double       divergence = 0.;
        const double step       = 1e-6;
        for (unsigned int d = 0; d < dim; ++d)
        {
          Tensor<1, dim> dx;
          dx[d] = step;
          const CahnHilliard::ReconstructedProfile<dim> plus{
            s + g * dx + 0.5 * (dx * (H * dx)), g + H * dx, H};
          const CahnHilliard::ReconstructedProfile<dim> minus{
            s - g * dx + 0.5 * (dx * (H * dx)), g - H * dx, H};
          divergence +=
            (CahnHilliard::evaluate_profile_correction(plus, epsilon).flux[d] -
             CahnHilliard::evaluate_profile_correction(minus, epsilon)
               .flux[d]) /
            (2. * step);
        }
        AssertThrow(std::abs(divergence - result.divergence) <
                      2e-6 * std::max(scale, std::abs(result.divergence)),
                    ExcMessage("Incorrect factorized spatial divergence."));

        for (unsigned int direction = 0; direction < 1 + dim + dim * dim;
             ++direction)
        {
          const double   ds = direction == 0 ? 1. : 0.;
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
          const auto analytic = jacobian.variation(ds, dg, dh);
          const CahnHilliard::ReconstructedProfile<dim> plus{s + step * ds,
                                                             g + step * dg,
                                                             H + step * dh};
          const CahnHilliard::ReconstructedProfile<dim> minus{s - step * ds,
                                                              g - step * dg,
                                                              H - step * dh};
          const auto                                    fp =
            CahnHilliard::evaluate_profile_correction(plus, epsilon);
          const auto fm =
            CahnHilliard::evaluate_profile_correction(minus, epsilon);
          AssertThrow(
            ((fp.flux - fm.flux) / (2. * step) - analytic.flux).norm() <
                2e-6 * std::max(scale, analytic.flux.norm()) &&
              std::abs((fp.divergence - fm.divergence) / (2. * step) -
                       analytic.divergence) <
                2e-6 * std::max(scale, std::abs(analytic.divergence)),
            ExcMessage(
              "Incorrect factorized Jacobian, including in the tails."));
        }
        // The Galerkin-only path must retain the same flux and flux Jacobian.
        CahnHilliard::ProfileCorrectionLinearization<dim> weak_jacobian;
        const auto weak = CahnHilliard::evaluate_profile_correction(
          profile, epsilon, false, &weak_jacobian);
        AssertThrow(
          (weak.flux - result.flux).norm() == 0. &&
            (weak_jacobian.flux_gradient - jacobian.flux_gradient).norm() ==
              0. &&
            (weak_jacobian.flux_value - jacobian.flux_value).norm() == 0.,
          ExcMessage("Weak and strong profile fluxes disagree."));
      }

  // Equilibrium, compressed/stretched profiles, zero gradient, and full
  // floating-point underflow; beta regularizes only the inverse gradient.
  for (const double s : {-400., -50., -8., 0., 8., 50., 400.})
    for (const double width_factor : {0., 0.5, 1., 2.})
    {
      constexpr double epsilon = 0.004;
      Tensor<1, dim>   g;
      g[0] = width_factor / (std::sqrt(2.) * epsilon);
      const CahnHilliard::ReconstructedProfile<dim>     profile{s, g, {}};
      CahnHilliard::ProfileCorrectionLinearization<dim> jacobian;
      const auto result = CahnHilliard::evaluate_profile_correction(profile,
                                                                    epsilon,
                                                                    true,
                                                                    &jacobian);
      AssertThrow(std::isfinite(result.flux.norm()) &&
                    std::isfinite(result.divergence) &&
                    std::isfinite(jacobian.divergence_value) &&
                    std::isfinite(jacobian.flux_gradient.norm()),
                  ExcMessage(
                    "Pure/flat phases must not produce NaN or infinity."));
      if (width_factor == 1.)
        AssertThrow(result.flux.norm() <= 1e-10 * profile.phase_factor &&
                      std::abs(result.divergence) <=
                        1e-8 * profile.phase_factor,
                    ExcMessage("Equilibrium tails must not be corrected."));
      else if (std::abs(profile.value) < CahnHilliard::profile_correction_cutoff &&
               width_factor > 0.)
        AssertThrow(result.flux[0] * (width_factor - 1.) > 0.,
                    ExcMessage("Profile correction has the wrong direction."));
      else
        AssertThrow(result.flux.norm() == 0. && result.divergence == 0.,
                    ExcMessage(
                      "Uniform pure phases must have zero correction."));
    }
}

// Check the chain rule through the finite inverse-tanh continuation using
// actual nodal phase perturbations, including pure values and overshoots.
void check_nodal_tail_variation()
{
  for (const double phase : {0.9996, 0.99975, 0.9999, 1., 1.0001, 1.01})
    for (const double sign : {-1., 1.})
    {
      const double left = sign * phase;
      const auto right = CahnHilliard::inverse_profile(sign * (phase - 0.0001));
      const auto evaluate = [&](const double value) {
        const auto   inverse = CahnHilliard::inverse_profile(value);
        Tensor<1, 2> g;
        g[0] = right.value - inverse.value;
        return CahnHilliard::ReconstructedProfile<2>{
          0.4 * inverse.value + 0.6 * right.value, g, {}};
      };
      const auto                                      profile = evaluate(left);
      CahnHilliard::ProfileCorrectionLinearization<2> jacobian;
      CahnHilliard::evaluate_profile_correction(profile,
                                                0.004,
                                                true,
                                                &jacobian);
      const double derivative = CahnHilliard::inverse_profile(left).derivative;
      Tensor<1, 2> dg;
      dg[0]                 = -derivative;
      const auto   analytic = jacobian.variation(0.4 * derivative, dg, {});
      const double step     = 1e-9;
      const auto   plus =
        CahnHilliard::evaluate_profile_correction(evaluate(left + step), 0.004);
      const auto minus =
        CahnHilliard::evaluate_profile_correction(evaluate(left - step), 0.004);
      AssertThrow(
        ((plus.flux - minus.flux) / (2. * step) - analytic.flux).norm() <
            2e-6 * std::max(profile.phase_factor, analytic.flux.norm()) &&
          std::abs((plus.divergence - minus.divergence) / (2. * step) -
                   analytic.divergence) <
            2e-6 *
              std::max(profile.phase_factor, std::abs(analytic.divergence)),
        ExcMessage("Incorrect nodal Jacobian near pure/overshooting phases."));
    }
}

int main()
{
  initlog();
  // A pure transported phase must reconstruct into the inactive region,
  // including its linear response to a small inverse-gradient perturbation.
  for (const double phi : {-1., 1.})
  {
    const CahnHilliard::ReconstructedProfile<2> profile{
      CahnHilliard::inverse_profile(phi).value, {}, {}};
    CahnHilliard::ProfileCorrectionLinearization<2> jacobian;
    const auto result = CahnHilliard::evaluate_profile_correction(
      profile, 0.004, true, &jacobian);
    AssertThrow(std::abs(profile.value) > CahnHilliard::profile_correction_cutoff &&
                  result.flux.norm() == 0. && result.divergence == 0. &&
                  jacobian.flux_gradient.norm() == 0.,
                ExcMessage("Pure phases must have no correction or antidiffusion."));
  }
  // An over-wide tail remains corrected inside the original activation
  // interval, on either side of the interface.
  for (const double phi : {-0.9996, 0.9996})
  {
    Tensor<1, 2> gradient;
    gradient[0]       = 0.5 * (1. - phi * phi) / (std::sqrt(2.) * 0.3);
    const auto result = CahnHilliard::evaluate_profile_correction(
      phi, gradient, Tensor<2, 2>(), 0.3);
    AssertThrow(result.flux[0] < 0.,
                ExcMessage(
                  "Profile correction must remain active in the tails."));
  }
  check_inverse_profile();
  check_reconstruction<2>();
  check_reconstruction<3>();
  check_flux<2>();
  check_flux<3>();
  check_linearization<2>();
  check_linearization<3>();
  check_factorized_correction<2>();
  check_factorized_correction<3>();
  check_nodal_tail_variation();
  deallog << "Profile flux: divergence, equilibrium, tails and zero gradient OK"
          << std::endl;
}
