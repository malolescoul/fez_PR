#ifndef chns_energy_h
#define chns_energy_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

/** Diagnostics for the standard CHNS equations. These integrals do not impose
 * an energy-consistent time discretization or an energy decay law. The clipped
 * density below is a reported reference; material properties remain unchanged.
 */
namespace CHNSEnergy
{
  inline double positive_density(double phi, double rho0, double rho1)
  {
    return 0.5 * ((1. + std::clamp(phi, -1., 1.)) * rho0 +
                  (1. - std::clamp(phi, -1., 1.)) * rho1);
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
    return R;
  }

  template <int dim, typename Scratch>
  double phase_residual(const Scratch &s, unsigned int q)
  {
    auto a = s.present_velocity_values[q];
    if constexpr (Scratch::enable_pseudo_solid)
      a -= s.present_mesh_velocity_values[q];
    return s.tracer_time_derivatives[q] + a * s.tracer_gradients[q] -
           s.get_phase_diffusion(q) + s.source_term_tracer[q];
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
      FluxProjectionPower,
      Count
    };
    std::array<double, Count> values{};
    double                    total_energy() const
    {
      return values[Kinetic] + values[Bulk] + values[Gradient] +
             values[Gravity];
    }
    double stabilization_power() const
    {
      return values[SUPGPower] + values[PSPGPower] + values[TracerSUPGPower];
    }
  };


} // namespace CHNSEnergy
#endif
