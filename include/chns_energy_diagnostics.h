#ifndef chns_energy_diagnostics_h
#define chns_energy_diagnostics_h
#include <chns_energy.h>

namespace CHNSEnergy
{
  template <int dim, typename Scratch>
  void integrate_point(const Scratch                &s,
                       unsigned int                  q,
                       const dealii::Tensor<1, dim> &phase_gradient_rate,
                       const dealii::Point<dim>     &point,
                       bool                          evaluate_power,
                       Integrals                    &data)
  {
    using I = Integrals;
    std::array<double, I::Count> v{};
    const double phi = s.tracer_values[q], mu = s.potential_values[q],
                 rho   = s.density[q];
    const auto  &u     = s.present_velocity_values[q];
    const auto  &gp    = s.tracer_gradients[q];
    const auto  &gm    = s.potential_gradients[q];
    auto         a     = u;
    double       div_w = 0.;
    if constexpr (Scratch::enable_pseudo_solid)
    {
      a -= s.present_mesh_velocity_values[q];
      if (s.skew_symmetric_momentum)
        div_w = trace(s.present_mesh_velocity_gradients[q]);
    }
    const double beta       = 0.5 * (s.density0 - s.density1);
    const double affine_rho = 0.5 * (s.density0 + s.density1) + beta * phi;
    const double well       = s.sigma_tilde / s.epsilon,
                 gamma      = s.sigma_tilde * s.epsilon;
    v[I::Volume]            = 1.;
    v[I::Phase]             = phi;
    v[I::AffineMass]        = affine_rho;
    v[I::PositiveMass]      = positive_density(phi, s.density0, s.density1);
    v[I::Kinetic]           = 0.5 * s.inertial_density[q] * u.norm_square();
    v[I::Bulk]              = 0.25 * well * (phi * phi - 1.) * (phi * phi - 1.);
    v[I::Gradient]          = 0.5 * gamma * gp.norm_square();
    v[I::Gravity]           = -affine_rho * (s.body_force * point);
    v[I::Viscous]   = 2. * s.dynamic_viscosity[q] *
                      scalar_product(s.present_velocity_sym_gradients[q],
                                     s.present_velocity_sym_gradients[q]);
    v[I::Diffusive] = s.get_mobility(q) * gm.norm_square();
    if (!s.profile_correction_flux.empty())
      v[I::CorrectionPower] = s.profile_correction_flux[q] * gm;
    v[I::BodyPower] = rho * (s.body_force * u);
    v[I::DivergenceSquared] =
      s.present_velocity_divergence[q] * s.present_velocity_divergence[q];
    v[I::PhaseOvershootVolume] = std::abs(phi) > 1. ? 1. : 0.;
    v[I::DensityExtensionL1]   = std::abs(v[I::PositiveMass] - affine_rho);
    const unsigned int phase_offset =
      phi >= 0. ? I::Phase0Kinetic : I::Phase1Kinetic;
    for (unsigned int k = 0; k < 4; ++k)
      v[phase_offset + k] = v[I::Kinetic + k];
    if (evaluate_power)
    {
      const double phidt = s.tracer_time_derivatives[q];
      const auto   inertia =
        s.skew_symmetric_momentum ?
          s.energy_time[q].inertia :
          s.inertial_density[q] * s.present_velocity_time_derivatives[q];
      const double potential = s.skew_symmetric_momentum ?
                                 s.energy_time[q].potential_derivative :
                                 phi * (phi * phi - 1.);
      v[I::TemporalPower]    = inertia * u + well * potential * phidt +
                               gamma * (gp * phase_gradient_rate);
      v[I::SourcePower] =
        s.source_term_velocity[q] * u + s.source_term_tracer[q] * mu -
        s.source_term_potential[q] * phidt -
        s.source_term_pressure[q] * s.present_pressure_values[q];
      if (s.skew_symmetric_momentum)
        v[I::TransportPower] = 0.5 * rho * div_w * u.norm_square() +
                               phi * ((u - a) * gm + div_w * mu);
      else
        v[I::TransportPower] =
          u * (s.present_velocity_gradients[q] *
               (rho * a + s.diffusive_flux_factor * s.get_phase_flux(q))) +
          mu * (a * gp) + phi * (u * gm);
      if (s.enable_stabilization)
      {
        const auto R = momentum_residual<dim>(s, q);
        const auto b = a + s.diffusive_flux_factor / rho * s.get_phase_flux(q);
        v[I::SUPGPower] =
          s.tau_supg_velocity[q] * (R * (s.present_velocity_gradients[q] * b));
        // The pressure test is -p, cancelling -(p,div u) in momentum.
        v[I::PSPGPower] =
          s.tau_supg_velocity[q] / rho * (R * s.present_pressure_gradients[q]);
      }
      if (s.enable_tracer_stabilization)
        v[I::TracerSUPGPower] =
          s.tau_supg_tracer[q] * phase_residual<dim>(s, q) * (a * gm);
    }
    for (unsigned int k = 0; k < I::Count; ++k)
      data.values[k] += s.JxW_moving[q] * v[k];
  }

} // namespace CHNSEnergy
#endif
