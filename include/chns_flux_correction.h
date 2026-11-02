#ifndef CHNS_FLUX_CORRECTION_H
#define CHNS_FLUX_CORRECTION_H

#include <cahn_hilliard.h>

namespace CahnHilliard
{
  /** Remove the chemical flux along the weighted reconstructed normal.
   * This object stores only the increment -M*n_delta*(n_delta.grad(mu));
   * M*grad(mu) and the profile-restoring flux are added separately.
   * Cache the scalar coefficient and its derivatives once per quadrature point
   * so all trial functions can reuse them during matrix assembly.
   */
  template <int dim>
  class FluxProjection
  {
  public:
    FluxProjection() = default;

    FluxProjection(const double                  phi,
                   const dealii::Tensor<1, dim> &gradient,
                   const dealii::Tensor<2, dim> &hessian,
                   const dealii::Tensor<1, dim> &potential_gradient,
                   const dealii::Tensor<2, dim> &potential_hessian,
                   const double                  mobility,
                   const dealii::Tensor<1, dim> &mobility_gradient,
                   const double                  epsilon,
                   const bool                    compute_divergence)
      : g(gradient)
      , v(potential_gradient)
      , grad_M(mobility_gradient)
      , H(hessian)
      , T(potential_hessian)
      , M(mobility)
      , with_divergence(compute_divergence)
    {
      const double tail = profile_correction_tail;
      const auto   a    = transition(1. - phi * phi, tail / 4., tail / 2.);
      if (a.value == 0.)
        return;
      s                  = g.norm_square();
      const double l     = std::sqrt(s);
      const double lower = tail / (4. * std::sqrt(2.) * epsilon);
      const auto   chi   = transition(l, lower, 2. * lower);
      // Both activations vanish with their first two derivatives. Return
      // before dividing by the gradient norm, including in pure phases.
      if (chi.value == 0.)
        return;
      const double ap  = -2. * phi * a.first;
      const double app = 4. * phi * phi * a.second - 2. * a.first;
      const double il = 1. / l, il2 = il * il, il3 = il2 * il;
      const double il4 = il2 * il2, il5 = il4 * il, il6 = il3 * il3;
      const double radial = chi.value * chi.value * il2;
      const double radial_s =
        chi.value * chi.first * il3 - chi.value * chi.value * il4;
      b    = a.value * a.value * radial;
      br   = 2. * a.value * ap * radial;
      bs   = a.value * a.value * radial_s;
      z    = g * v;
      flux = -M * b * z * g;
      if (!with_divergence)
        return;
      const double radial_ss =
        0.5 * (chi.first * chi.first + chi.value * chi.second) * il4 -
        2.5 * chi.value * chi.first * il5 + 2. * chi.value * chi.value * il6;
      brr    = 2. * (ap * ap + a.value * app) * radial;
      brs    = 2. * a.value * ap * radial_s;
      bss    = a.value * a.value * radial_ss;
      gHg    = g * (H * g);
      gHv    = g * (H * v);
      gTg    = g * (T * g);
      gradMg = grad_M * g;
      // Write the projection as -M*b(r,|grad(r)|^2)*(grad(r).grad(mu))*grad(r).
      // L accounts for grad(b) and div(grad(r)); D also differentiates
      // grad(mu).
      L          = br * s + 2. * bs * gHg + b * trace(H);
      D          = L * z + b * (gHv + gTg);
      divergence = -b * z * gradMg - M * D;
    }

    ProfileCorrection<dim>
    variation(const ProfileVariation<dim>  &profile,
              const dealii::Tensor<1, dim> &dv,
              const dealii::Tensor<2, dim> &dT,
              const double                  dM,
              const dealii::Tensor<1, dim> &dgrad_M) const
    {
      if (b == 0.)
        return {};
      const auto            &dg = profile.gradient;
      const auto            &dH = profile.hessian;
      const double           ds = 2. * g * dg;
      const double           dz = dg * v + g * dv;
      const double           db = br * profile.value + bs * ds;
      ProfileCorrection<dim> result;
      result.flux =
        -(dM * b * z + M * db * z + M * b * dz) * g - M * b * z * dg;
      if (!with_divergence)
        return result;
      const double dbr  = brr * profile.value + brs * ds;
      const double dbs  = brs * profile.value + bss * ds;
      const double dgHg = dg * (H * g) + g * (dH * g) + g * (H * dg);
      const double dgHv = dg * (H * v) + g * (dH * v) + g * (H * dv);
      const double dgTg = dg * (T * g) + g * (dT * g) + g * (T * dg);
      const double dL   = dbr * s + br * ds + 2. * dbs * gHg + 2. * bs * dgHg +
                        db * trace(H) + b * trace(dH);
      const double dD = dL * z + L * dz + db * (gHv + gTg) + b * (dgHv + dgTg);
      result.divergence = -db * z * gradMg - b * dz * gradMg -
                          b * z * (dgrad_M * g + grad_M * dg) - dM * D - M * dD;
      return result;
    }

    dealii::Tensor<1, dim> flux;
    double                 divergence = 0.;

  private:
    struct Transition
    {
      double value, first, second;
    };

    static Transition
    transition(const double value, const double lower, const double upper)
    {
      if (value <= lower)
        return {0., 0., 0.};
      if (value >= upper)
        return {1., 0., 0.};
      const double width = upper - lower;
      const double t     = (value - lower) / width;
      return {t * t * t * (10. + t * (-15. + 6. * t)),
              30. * t * t * (t - 1.) * (t - 1.) / width,
              60. * t * (2. * t * t - 3. * t + 1.) / (width * width)};
    }

    dealii::Tensor<1, dim> g, v, grad_M;
    dealii::Tensor<2, dim> H, T;
    double                 M               = 0.;
    bool                   with_divergence = false;
    double b = 0., br = 0., bs = 0., brr = 0., brs = 0., bss = 0.;
    double s = 0., z = 0., gHg = 0., gHv = 0., gTg = 0., gradMg = 0.;
    double L = 0., D = 0.;
  };
} // namespace CahnHilliard

#endif
