#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

namespace CahnHilliard
{
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
