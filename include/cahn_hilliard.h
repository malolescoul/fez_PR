#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

#include <cmath>

namespace CahnHilliard
{
  struct ProfileInverse
  {
    double value;
    double derivative;
  };

  /** Inverse of the tanh profile, atanh(phi), with an odd C2 continuation
   * beyond |phi| = sqrt(1 - (1 - 0.999^2) / 4). The continuation has a bounded
   * positive slope, so pure phases and overshoots give finite values without
   * clipping phi. Used to measure the distance to the interface.
   */
  inline ProfileInverse inverse_profile(const double phi)
  {
    const double cutoff = std::sqrt(1. - 0.25 * (1. - 0.999 * 0.999));
    if (std::abs(phi) <= cutoff)
      return {std::atanh(phi), 1. / (1. - phi * phi)};
    const double slope  = 1. / (1. - cutoff * cutoff);
    const double length = (1. - cutoff * cutoff) / cutoff;
    const double offset = std::abs(phi) - cutoff;
    return {std::copysign(std::atanh(cutoff) +
                            slope *
                              (3. * offset +
                               2. * length * std::expm1(-offset / length)),
                          phi),
            slope * (3. - 2. * std::exp(-offset / length))};
  }

  template <int dim>
  struct ProfileCorrection
  {
    dealii::Tensor<1, dim> flux;
    double                 divergence = 0.;
  };

  /** Derivatives of the profile flux and its divergence at one quadrature
   * point. The global mobility amplitude is held constant during linearization.
   */
  template <int dim>
  struct ProfileCorrectionLinearization
  {
    dealii::Tensor<1, dim> flux_phi;
    dealii::Tensor<2, dim> flux_gradient;
    double                 divergence_phi = 0.;
    dealii::Tensor<1, dim> divergence_gradient;

    ProfileCorrection<dim>
    variation(const double                  dphi,
              const dealii::Tensor<1, dim> &dgradient,
              const dealii::Tensor<2, dim> &dhessian) const
    {
      return {dphi * flux_phi + flux_gradient * dgradient,
              divergence_phi * dphi + divergence_gradient * dgradient +
                scalar_product(flux_gradient, dhessian)};
    }
  };

  /** Gradient floor of the profile flux, in 1/length units.
   * Fixed internal numerical scale; there is no user-facing parameter.
   */
  inline double profile_correction_gradient_floor(const double epsilon)
  {
    return 1e-4 / epsilon;
  }

  /**
   * Regularized profile flux F = (1-Q/R)*grad(phi), where
   * q=(1-phi^2)/(sqrt(2)*epsilon), R^2=|grad(phi)|^2+beta^2*q^2+delta^2 and
   * Q^2=(1+beta^2)*q^2+delta^2. Thus F vanishes whenever |grad(phi)|=|q|,
   * in particular for the equilibrium tanh profile, and in uniform phases.
   * beta regularizes the normal where the phase gradient vanishes, and delta is
   * the gradient floor above: the anti-diffusive slope 1-Q/R at grad(phi)=0
   * tends to zero in the pure phases instead of the q-independent value
   * 1-sqrt(1+beta^2)/beta. Beyond |phi|=1, q is replaced by zero: the flux is
   * purely diffusive there, so overshoots are never sharpened. As only q^2
   * enters, F and its divergence are continuous across |phi|=1.
   * The caller supplies the global mobility amplitude separately.
   * When requested, compute the derivatives once for all trial functions.
   */
  template <int dim>
  ProfileCorrection<dim> evaluate_profile_correction(
    const double                         phi,
    const dealii::Tensor<1, dim>        &gradient,
    const dealii::Tensor<2, dim>        &hessian,
    const double                         epsilon,
    const bool                           compute_divergence = true,
    ProfileCorrectionLinearization<dim> *linearization      = nullptr)
  {
    if (linearization)
      *linearization = {};
    constexpr double beta     = 0.05;
    constexpr double c2       = 1. + beta * beta;
    const double     delta    = profile_correction_gradient_floor(epsilon);
    const bool       interior = phi * phi < 1.;
    const double     q =
      interior ? (1. - phi * phi) / (std::sqrt(2.) * epsilon) : 0.;
    const double s = gradient.norm_square();
    const double R = std::sqrt(s + beta * beta * q * q + delta * delta);
    const double Q = std::sqrt(c2 * q * q + delta * delta);
    const double f = 1. - Q / R;
    ProfileCorrection<dim> result;
    result.flux = f * gradient;
    if (!compute_divergence && !linearization)
      return result;

    // With F = f(phi,s)*grad(phi), s=|grad(phi)|^2, its divergence is
    // f*tr(H) + f_phi*s + 2*f_s*grad(phi).H.grad(phi).
    // u=q*dq/dphi; Q_phi=c2*u/Q and R_phi=beta^2*u/R.
    const double dq     = interior ? -2. * phi / (std::sqrt(2.) * epsilon) : 0.;
    const double u      = q * dq;
    const double Q_phi  = c2 * u / Q;
    const double R_phi  = beta * beta * u / R;
    const double inv_R3 = 1. / (R * R * R);
    const double f_phi  = -Q_phi / R + Q * R_phi / (R * R);
    const double f_s    = 0.5 * Q * inv_R3;
    const auto   Hg     = hessian * gradient;
    const double gHg    = gradient * Hg;
    if (compute_divergence)
      result.divergence = f * trace(hessian) + f_phi * s + 2. * f_s * gHg;
    if (linearization)
    {
      linearization->flux_phi = f_phi * gradient;
      linearization->flux_gradient =
        2. * f_s * outer_product(gradient, gradient);
      for (unsigned int d = 0; d < dim; ++d)
        linearization->flux_gradient[d][d] += f;
      if (compute_divergence)
      {
        // Only the second phi-derivative jumps at |phi|=1.
        const double ddq    = interior ? -2. / (std::sqrt(2.) * epsilon) : 0.;
        const double inv_R5 = inv_R3 / (R * R);
        const double du     = dq * dq + q * ddq;
        const double Q_pp   = c2 * du / Q - c2 * c2 * u * u / (Q * Q * Q);
        const double R_pp =
          beta * beta * du / R - std::pow(beta, 4) * u * u * inv_R3;
        const double f_pp = -Q_pp / R + 2. * Q_phi * R_phi / (R * R) +
                            Q * R_pp / (R * R) -
                            2. * Q * R_phi * R_phi * inv_R3;
        const double f_ps = 0.5 * Q_phi * inv_R3 - 1.5 * Q * R_phi * inv_R3 / R;
        const double f_ss = -0.75 * Q * inv_R5;
        linearization->divergence_phi =
          f_phi * trace(hessian) + f_pp * s + 2. * f_ps * gHg;
        linearization->divergence_gradient =
          (2. * f_s * trace(hessian) + 2. * f_ps * s + 2. * f_phi +
           4. * f_ss * gHg) *
            gradient +
          4. * f_s * Hg;
      }
    }
    return result;
  }

  /**
   * Simply return the passed phase marker
   */
  inline double tracer_identity(const double phase_marker)
  {
    return phase_marker;
  }

  /**
   * Apply a limiter to the phase tracer : res = max(-1, min(1, phi))
   */
  inline double tracer_limiter(const double phase_marker)
  {
    return std::max(-1., std::min(1., phase_marker));
  }

  /**
   * Return a pointer to the limiter function used for the phase field tracer
   */
  using TracerLimiterFunction = double (*)(double);

  template <int dim>
  TracerLimiterFunction
  get_limiter_function(const Parameters::CahnHilliard<dim> param)
  {
    if (param.with_tracer_limiter)
      return &tracer_limiter;
    else
      return &tracer_identity;
  }

  /**
   * Apply linear mixing from value A (when phase marker = 1) to value B (phase
   * marker = -1).
   */
  inline double linear_mixing(const double phase_marker,
                              const double val_a,
                              const double val_b)
  {
    return 0.5 * ((val_a - val_b) * phase_marker + (val_a + val_b));
  }

  /**
   * Derivative w.r.t. the tracer of the linear mixing function
   */
  inline double linear_mixing_derivative(const double /*phase_marker*/,
                                         const double val_a,
                                         const double val_b)
  {
    return 0.5 * (val_a - val_b);
  }
  /** Mobility and its partial derivatives at fixed physical velocity and
   * tracer gradient. The adaptive laws use u, including on moving meshes;
   * the relative velocity u-w is used only in the transport equations. */
  template <int dim>
  struct MobilityEvaluation
  {
    double                 value;
    double                 derivative_wrt_tracer        = 0.;
    double                 second_derivative_wrt_tracer = 0.;
    dealii::Tensor<1, dim> derivative_wrt_velocity;
    dealii::Tensor<1, dim> derivative_wrt_tracer_gradient;
  };

  struct AdaptiveMobilityScaling
  {
    double coefficient = 0.;
    double delta       = 0.;
  };

  template <int dim>
  inline bool
  is_adaptive_mobility_model(const Parameters::CahnHilliard<dim> &param)
  {
    using Model = typename Parameters::CahnHilliard<dim>::MobilityModel;
    return param.mobility_model == Model::adaptive_normal_velocity ||
           param.mobility_model == Model::adaptive_velocity;
  }

  template <int dim>
  inline AdaptiveMobilityScaling
  get_adaptive_mobility_scaling(const Parameters::CahnHilliard<dim> &param)
  {
    if (!is_adaptive_mobility_model(param))
      return {};
    AssertThrow(param.surface_tension > 0. && param.epsilon_interface > 0.,
                dealii::ExcMessage(
                  "Adaptive mobility requires positive surface "
                  "tension and interface thickness."));
    const double epsilon = param.epsilon_interface;
    const double sigma_tilde =
      3. / (2. * std::sqrt(2.)) * param.surface_tension;
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::adaptive_normal_velocity)
      return {param.mobility_scaling_factor * std::sqrt(2.) * epsilon *
                epsilon * epsilon / sigma_tilde,
              param.mobility_regularization};
    return {param.mobility_scaling_factor * epsilon * epsilon / sigma_tilde,
            param.velocity_regularization};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_constant_mobility(const Parameters::CahnHilliard<dim> &param,
                             const double,
                             const dealii::Tensor<1, dim> &,
                             const dealii::Tensor<1, dim> &,
                             const AdaptiveMobilityScaling &)
  {
    return {param.mobility, 0., 0., {}, {}};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_adaptative_mobility(const Parameters::CahnHilliard<dim> &,
                               const double,
                               const dealii::Tensor<1, dim>  &velocity,
                               const dealii::Tensor<1, dim>  &tracer_gradient,
                               const AdaptiveMobilityScaling &scaling)
  {
    const double raw = scaling.coefficient * (velocity * tracer_gradient);
    const double regularized = std::hypot(raw, scaling.delta);
    const double sensitivity = scaling.coefficient * raw / regularized;
    return {regularized,
            0.,
            0.,
            sensitivity * tracer_gradient,
            sensitivity * velocity};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_adaptative_mobility_3(const Parameters::CahnHilliard<dim> &,
                                 const double,
                                 const dealii::Tensor<1, dim> &velocity,
                                 const dealii::Tensor<1, dim> &,
                                 const AdaptiveMobilityScaling &scaling)
  {
    const double norm = std::hypot(velocity.norm(), scaling.delta);
    return {scaling.coefficient * norm,
            0.,
            0.,
            scaling.coefficient / norm * velocity,
            {}};
  }

  /** The expression uses x as the phase value, not as a spatial coordinate.
   * Outside [-1,1], the optional clamp has zero first and second derivatives.
   * At its endpoints we select the clamped-side derivatives. */
  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_degenerate_mobility(const Parameters::CahnHilliard<dim> &param,
                               const double                         phi,
                               const dealii::Tensor<1, dim> &,
                               const dealii::Tensor<1, dim> &,
                               const AdaptiveMobilityScaling &)
  {
    dealii::Point<dim> point;
    point[0] = param.mobility_tracer_limiter ? tracer_limiter(phi) : phi;
    const double value = param.degenerate_mobility->value(point);
    if (param.mobility_tracer_limiter && std::abs(phi) >= 1.)
      return {value, 0., 0., {}, {}};
    return {value,
            param.degenerate_mobility->gradient(point)[0],
            param.degenerate_mobility->hessian(point)[0][0],
            {},
            {}};
  }

  template <int dim>
  using MobilityEvaluationFunction =
    MobilityEvaluation<dim> (*)(const Parameters::CahnHilliard<dim> &,
                                double,
                                const dealii::Tensor<1, dim> &,
                                const dealii::Tensor<1, dim> &,
                                const AdaptiveMobilityScaling &);

  template <int dim>
  MobilityEvaluationFunction<dim>
  get_mobility_evaluation_function(const Parameters::CahnHilliard<dim> &param)
  {
    using Model = typename Parameters::CahnHilliard<dim>::MobilityModel;
    switch (param.mobility_model)
    {
      case Model::constant:
        return &evaluate_constant_mobility<dim>;
      case Model::degenerate:
        return &evaluate_degenerate_mobility<dim>;
      case Model::adaptive_normal_velocity:
        return &evaluate_adaptative_mobility<dim>;
      case Model::adaptive_velocity:
        return &evaluate_adaptative_mobility_3<dim>;
    }
    AssertThrow(false, dealii::ExcMessage("Unknown mobility model."));
    return nullptr;
  }

  /** Hessian blocks needed to differentiate grad(M) in the strong diffusion
   * residual. The mixed block is d(dM/du)/d(grad(phi)). */
  template <int dim>
  struct AdaptiveMobilityHessians
  {
    dealii::Tensor<2, dim> velocity_velocity;
    dealii::Tensor<2, dim> velocity_gradient;
    dealii::Tensor<2, dim> gradient_gradient;
  };

  template <int dim>
  inline AdaptiveMobilityHessians<dim> evaluate_adaptive_mobility_hessians(
    const Parameters::CahnHilliard<dim> &param,
    const dealii::Tensor<1, dim>        &velocity,
    const dealii::Tensor<1, dim>        &tracer_gradient,
    const AdaptiveMobilityScaling       &scaling)
  {
    using Model = typename Parameters::CahnHilliard<dim>::MobilityModel;
    AdaptiveMobilityHessians<dim> result;
    if (param.mobility_model == Model::adaptive_velocity)
    {
      const double norm = std::hypot(velocity.norm(), scaling.delta);
      const auto   normalized_velocity = velocity / norm;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          result.velocity_velocity[i][j] =
            scaling.coefficient / norm *
            ((i == j ? 1. : 0.) -
             normalized_velocity[i] * normalized_velocity[j]);
      return result;
    }
    AssertThrow(param.mobility_model == Model::adaptive_normal_velocity,
                dealii::ExcMessage(
                  "Adaptive mobility Hessians require model 1 or 3."));
    const double raw    = scaling.coefficient * (velocity * tracer_gradient);
    const double radius = std::hypot(raw, scaling.delta);
    const double sensitivity = scaling.coefficient * raw / radius;
    const double ratio       = scaling.delta / radius;
    const double curvature =
      scaling.coefficient * scaling.coefficient / radius * ratio * ratio;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
      {
        result.velocity_velocity[i][j] =
          curvature * tracer_gradient[i] * tracer_gradient[j];
        result.velocity_gradient[i][j] =
          curvature * tracer_gradient[i] * velocity[j] +
          (i == j ? sensitivity : 0.);
        result.gradient_gradient[i][j] = curvature * velocity[i] * velocity[j];
      }
    return result;
  }
} // namespace CahnHilliard

#endif
