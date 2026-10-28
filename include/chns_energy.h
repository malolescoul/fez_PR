#ifndef chns_energy_h
#define chns_energy_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

/** Temporal terms adapted from the Brunk--ten Eikelder discretization.
 * The affine Abels density is retained in transport; only kinetic energy uses
 * the positive extension below. BDF2's kinetic G-identity does not imply a
 * decay theorem for the complete nonlinear free energy.
 */
namespace CHNSEnergy
{
  inline double positive_density(double phi, double rho0, double rho1)
  {
    return 0.5 * ((1. + std::clamp(phi, -1., 1.)) * rho0 +
                  (1. - std::clamp(phi, -1., 1.)) * rho1);
  }

  template <int dim>
  struct TimeData
  {
    double                 density;
    dealii::Tensor<1, dim> inertia, inertia_phase_derivative;
    double                 inertia_velocity_derivative;
    double                 potential_derivative, potential_second_derivative;
  };

  template <int dim>
  TimeData<dim>
  evaluate_time(double                                       phi,
                const dealii::Tensor<1, dim>                &u,
                double                                       rho0,
                double                                       rho1,
                const std::vector<double>                   &c,
                const std::array<double, 2>                 &old_phi,
                const std::array<dealii::Tensor<1, dim>, 2> &old_u)
  {
    TimeData<dim> t;
    t.density         = positive_density(phi, rho0, rho1);
    const double drho = std::abs(phi) > 1. ? 0. : 0.5 * (rho0 - rho1);
    const bool   bdf2 = c.size() > 2 && c[2] != 0.;
    if (!bdf2)
    {
      const double old_rho = positive_density(old_phi[0], rho0, rho1);
      t.inertia =
        c[0] * (0.5 * (t.density - old_rho) * u + old_rho * (u - old_u[0]));
      t.inertia_velocity_derivative = 0.5 * c[0] * (t.density + old_rho);
      t.inertia_phase_derivative    = 0.5 * c[0] * drho * u;
      const double b                = old_phi[0];
      t.potential_derivative = 0.25 * (phi + b) * (phi * phi + b * b - 2.);
      t.potential_second_derivative =
        0.25 * (3. * phi * phi + 2. * phi * b + b * b - 2.);
    }
    else
    {
      const double sigma      = std::sqrt(t.density);
      auto         derivative = c[0] * sigma * u;
      for (unsigned int k = 1; k < c.size(); ++k)
        derivative += c[k] *
                      std::sqrt(positive_density(old_phi[k - 1], rho0, rho1)) *
                      old_u[k - 1];
      t.inertia                     = sigma * derivative;
      t.inertia_velocity_derivative = c[0] * t.density;
      t.inertia_phase_derivative =
        0.5 * drho / sigma * (derivative + c[0] * sigma * u);
      t.potential_derivative        = phi * (phi * phi - 1.);
      t.potential_second_derivative = 3. * phi * phi - 1.;
    }
    return t;
  }
  // These strong residuals are shared by assembly and power diagnostics.
  template <int dim, typename Scratch>
  dealii::Tensor<1, dim> momentum_residual(const Scratch &s, unsigned int q)
  {
    const auto &u = s.present_velocity_values[q];
    auto        a = u;
    if constexpr (Scratch::enable_pseudo_solid)
      a -= s.present_mesh_velocity_values[q];
    const double rho = s.density[q];
    auto         R =
      rho * (s.present_velocity_time_derivatives[q] +
             s.present_velocity_gradients[q] * a - s.body_force) +
      s.diffusive_flux[q] + s.tracer_values[q] * s.potential_gradients[q] +
      s.present_pressure_gradients[q] + s.source_term_velocity[q] -
      s.dynamic_viscosity[q] *
        (s.present_velocity_laplacians[q] + s.present_velocity_grad_div[q]) -
      2. * s.derivative_dynamic_viscosity_wrt_tracer[q] *
        (s.present_velocity_sym_gradients[q] * s.tracer_gradients[q]);
    if (s.skew_symmetric_momentum)
      R += s.energy_time[q].inertia -
           rho * s.present_velocity_time_derivatives[q] +
           0.5 *
             (s.derivative_density_wrt_tracer[q] * (s.tracer_gradients[q] * a) +
              rho * s.present_velocity_divergence[q] +
              s.diffusive_flux_factor * s.get_phase_diffusion(q)) *
             u;
    return R;
  }

  template <int dim, typename Scratch>
  double phase_residual(const Scratch &s, unsigned int q)
  {
    auto a = s.present_velocity_values[q];
    if constexpr (Scratch::enable_pseudo_solid)
      a -= s.present_mesh_velocity_values[q];
    return s.tracer_time_derivatives[q] + a * s.tracer_gradients[q] -
           s.get_phase_diffusion(q) + s.source_term_tracer[q] +
           (s.conservative_tracer_transport ?
              s.tracer_values[q] * s.present_velocity_divergence[q] :
              0.);
  }

  /** Domain integrals. Powers are signed contributions on the left of the
   * energy balance: a positive stabilization power removes energy.
   * Phase energies partition quadrature points by the sign of phi, as on the
   * source energy-monitoring branch (fluid 0 for phi >= 0).
   */
  struct Integrals
  {
    enum Index
    {
      Volume,
      Phase,
      AffineMass,
      PositiveMass,
      Kinetic,
      Bulk,
      Gradient,
      Gravity,
      Viscous,
      Diffusive,
      CorrectionPower,
      BodyPower,
      SourcePower,
      TemporalPower,
      TransportPower,
      SUPGPower,
      PSPGPower,
      TracerSUPGPower,
      DivergenceSquared,
      PhaseOvershootVolume,
      DensityExtensionL1,
      Phase0Kinetic,
      Phase0Bulk,
      Phase0Gradient,
      Phase0Gravity,
      Phase1Kinetic,
      Phase1Bulk,
      Phase1Gradient,
      Phase1Gravity,
      Count
    };
    std::array<double, Count> values{};
    double                    total_energy() const
    {
      return values[Kinetic] + values[Bulk] + values[Gradient] +
             values[Gravity];
    }
    double stabilization_power() const
    { return values[SUPGPower] + values[PSPGPower] + values[TracerSUPGPower]; }
  };


} // namespace CHNSEnergy
#endif
