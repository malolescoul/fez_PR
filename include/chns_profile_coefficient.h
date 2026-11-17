#ifndef CHNS_PROFILE_COEFFICIENT_H
#define CHNS_PROFILE_COEFFICIENT_H
#include <cahn_hilliard.h>
#include <chns_profile_geometry.h>

namespace CahnHilliard
{
  // Two independent differential directions: spatial x and Newton increment v.
  // xv is the mixed derivative, not a finite difference.
  struct ProfileCoefficientJet
  {
    double f = 0., x = 0., v = 0., xv = 0.;
    ProfileCoefficientJet() = default;
    ProfileCoefficientJet(double f,
                          double x  = 0.,
                          double v  = 0.,
                          double xv = 0.)
      : f(f)
      , x(x)
      , v(v)
      , xv(xv)
    {}
  };
  inline ProfileCoefficientJet operator+(ProfileCoefficientJet a,
                                         ProfileCoefficientJet b)
  {
    return {a.f + b.f, a.x + b.x, a.v + b.v, a.xv + b.xv};
  }
  inline ProfileCoefficientJet operator-(ProfileCoefficientJet a,
                                         ProfileCoefficientJet b)
  {
    return {a.f - b.f, a.x - b.x, a.v - b.v, a.xv - b.xv};
  }
  inline ProfileCoefficientJet operator*(ProfileCoefficientJet a,
                                         ProfileCoefficientJet b)
  {
    return {a.f * b.f,
            a.x * b.f + a.f * b.x,
            a.v * b.f + a.f * b.v,
            a.xv * b.f + a.x * b.v + a.v * b.x + a.f * b.xv};
  }
  inline ProfileCoefficientJet operator/(ProfileCoefficientJet a,
                                         ProfileCoefficientJet b)
  {
    const double r = 1. / b.f;
    return a * ProfileCoefficientJet(r,
                                     -b.x * r * r,
                                     -b.v * r * r,
                                     2. * b.x * b.v * r * r * r - b.xv * r * r);
  }
  /** Local amplitude, with constant mobility. Here base is
   * M*sigma_tilde/epsilon and w=grad(interpolated inverse profile).
   * The normal uses the same beta as the restoring flux. Smooth abs avoids
   * a Newton kink at zero normal strain; neither regularization is user tuned.
   */
  template <int dim>
  struct LocalProfileCoefficient
  {
    double                 value = 0.;
    dealii::Tensor<1, dim> gradient;

    LocalProfileCoefficient() = default;
    explicit LocalProfileCoefficient(const double constant)
      : value(constant)
    {}

    LocalProfileCoefficient(
      const dealii::Tensor<1, dim>          &w,
      const dealii::Tensor<2, dim>          &H,
      const dealii::Tensor<2, dim>          &U,
      const dealii::Tensor<3, dim>          &V,
      const double                           eps,
      const double                           base,
      const double                           C,
      const CHNSProfileOverlap::PenaltyValue geometry = {})
      : w(w)
      , H(H)
      , U(U)
      , V(V)
      , factor(C * eps * eps)
    {
      constexpr double beta = 0.05;
      denominator           = w * w + beta * beta / (2. * eps * eps);
      z                     = (w * (U * w)) / denominator;
      const double delta    = 1e-8 * base / (eps * eps);
      const double radius   = std::hypot(z, delta);
      first                 = z / radius;
      second                = (delta / radius) * (delta / radius) / radius;
      value                 = C * base + factor * z * (z / (radius + delta));
      for (unsigned k = 0; k < dim; ++k)
      {
        double numerator_gradient = 0.;
        for (unsigned i = 0; i < dim; ++i)
        {
          denominator_gradient[k] += 2. * w[i] * H[i][k];
          for (unsigned j = 0; j < dim; ++j)
            numerator_gradient += H[i][k] * U[i][j] * w[j] +
                                  w[i] * U[i][j] * H[j][k] +
                                  w[i] * V[i][j][k] * w[j];
        }
        z_gradient[k] =
          (numerator_gradient - z * denominator_gradient[k]) / denominator;
      }
      gradient        = factor * first * z_gradient;
      geometric       = geometry;
      profile_epsilon = eps;
      rate_value      = value;
      {
        for (unsigned k = 0; k < dim; ++k)
          gradient[k] = geometric.feedback.multiplier * gradient[k] +
                        value * geometric.feedback.gradient[k];
        value *= geometric.feedback.multiplier;
      }
      base_value    = value;
      base_gradient = gradient;
      for (unsigned k = 0; k < dim; ++k)
      {
        const ProfileCoefficientJet b(base_value, base_gradient[k]);
        const auto                  limited = attenuate(b, k);
        value                               = limited.f;
        gradient[k]                         = limited.x;
      }
    }

    LocalProfileCoefficient variation(const dealii::Tensor<1, dim> &dw,
                                      const dealii::Tensor<2, dim> &dH,
                                      const dealii::Tensor<2, dim> &dU,
                                      const dealii::Tensor<3, dim> &dV) const
    {
      LocalProfileCoefficient result;
      if (factor == 0.)
        return result;
      const double ddenominator = 2. * w * dw;
      const double dz =
        (dw * (U * w) + w * (U * dw) + w * (dU * w) - z * ddenominator) /
        denominator;
      result.value = factor * first * dz;
      for (unsigned k = 0; k < dim; ++k)
      {
        double dnumerator_gradient = 0., ddenominator_gradient = 0.;
        for (unsigned i = 0; i < dim; ++i)
        {
          ddenominator_gradient += 2. * (dw[i] * H[i][k] + w[i] * dH[i][k]);
          for (unsigned j = 0; j < dim; ++j)
            dnumerator_gradient +=
              dH[i][k] * U[i][j] * w[j] + H[i][k] * dU[i][j] * w[j] +
              H[i][k] * U[i][j] * dw[j] + dw[i] * U[i][j] * H[j][k] +
              w[i] * dU[i][j] * H[j][k] + w[i] * U[i][j] * dH[j][k] +
              dw[i] * V[i][j][k] * w[j] + w[i] * dV[i][j][k] * w[j] +
              w[i] * V[i][j][k] * dw[j];
        }
        const double dz_gradient =
          (dnumerator_gradient - dz * denominator_gradient[k] -
           z * ddenominator_gradient - z_gradient[k] * ddenominator) /
          denominator;
        result.gradient[k] =
          factor * (second * dz * z_gradient[k] + first * dz_gradient);
      }
      {
        for (unsigned k = 0; k < dim; ++k)
          result.gradient[k] =
            geometric.feedback.multiplier * result.gradient[k] +
            result.value * geometric.feedback.gradient[k];
        result.value *= geometric.feedback.multiplier;
      }
      // Geometry remains fixed; differentiate the local phase and velocity.
      const double dbase = result.value;
      for (unsigned k = 0; k < dim; ++k)
      {
        const ProfileCoefficientJet b(base_value,
                                      base_gradient[k],
                                      dbase,
                                      result.gradient[k]);
        const auto                  limited = attenuate(b, k, dw, dH);
        result.value                        = limited.v;
        result.gradient[k]                  = limited.xv;
      }
      return result;
    }

    double unattenuated_value() const { return base_value; }
    double timescale_value() const { return rate_value; }

    ProfileCorrection<dim> apply(const ProfileCorrection<dim> &flux) const
    {
      return {value * flux.flux,
              value * flux.divergence + gradient * flux.flux};
    }

    ProfileCorrection<dim>
    apply_variation(const ProfileCorrection<dim>  &flux,
                    const ProfileCorrection<dim>  &dflux,
                    const LocalProfileCoefficient &dcoefficient) const
    {
      const auto a = apply(dflux), b = dcoefficient.apply(flux);
      return {a.flux + b.flux, a.divergence + b.divergence};
    }

  private:
    dealii::Tensor<1, dim> w, denominator_gradient, z_gradient;
    dealii::Tensor<2, dim> H, U;
    dealii::Tensor<3, dim> V;
    double factor = 0., denominator = 1., z = 0., first = 0., second = 0.;
    double base_value = 0., rate_value = 0.;
    dealii::Tensor<1, dim>           base_gradient;
    CHNSProfileOverlap::PenaltyValue geometric;
    double                           profile_epsilon = 1.;
    ProfileCoefficientJet
    ratio_squared(const unsigned                k,
                  const dealii::Tensor<1, dim> &dw = {},
                  const dealii::Tensor<2, dim> &dH = {}) const
    {
      ProfileCoefficientJet result;
      for (unsigned i = 0; i < dim; ++i)
      {
        const ProfileCoefficientJet g(w[i], H[i][k], dw[i], dH[i][k]);
        result = result + 2. * profile_epsilon * profile_epsilon * g * g;
      }
      return result;
    }
    ProfileCoefficientJet attenuate(const ProfileCoefficientJet  &b,
                                    const unsigned                k,
                                    const dealii::Tensor<1, dim> &dw = {},
                                    const dealii::Tensor<2, dim> &dH = {}) const
    {
      const auto geometric_limit =
        b /
        (1. + ProfileCoefficientJet(geometric.value, geometric.gradient[k]));
      const auto compression = ratio_squared(k, dw, dH) - 1.;
      if (compression.f <= 0.)
        return geometric_limit;
      // C2 recovery of useful diffusion, with no finite full-strength
      // plateau that can mask an approaching interaction. Evaluate the
      // rational cubic in reciprocal form above one to avoid overflow.
      ProfileCoefficientJet restore;
      if (compression.f <= 1.)
      {
        const auto cube = compression * compression * compression;
        restore         = cube / (1. + cube);
      }
      else
      {
        const auto inverse = 1. / compression;
        restore            = 1. / (1. + inverse * inverse * inverse);
      }
      return geometric_limit + (b - geometric_limit) * restore;
    }
  };
} // namespace CahnHilliard
#endif
