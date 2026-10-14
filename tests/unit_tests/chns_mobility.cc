#include <cahn_hilliard.h>

#include "../tests.h"

template <int dim>
void check_derivatives()
{
  using Model = typename Parameters::CahnHilliard<dim>::MobilityModel;
  Parameters::CahnHilliard<dim> param;
  param.surface_tension         = 1.3;
  param.epsilon_interface       = 0.4;
  param.mobility_scaling_factor = 2.;
  param.mobility_regularization = 0.08;
  param.velocity_regularization = 0.08;
  Tensor<1, dim> u, g, direction;
  for (unsigned int d = 0; d < dim; ++d)
  {
    u[d]         = 0.3 + 0.2 * d;
    g[d]         = d == 0 ? -0.7 : 0.4;
    direction[d] = d == 0 ? 0.2 : -0.6;
  }
  direction /= direction.norm();
  for (const auto model :
       {Model::adaptive_normal_velocity, Model::adaptive_velocity})
    for (unsigned int state = 0; state < 4; ++state)
    {
      param.mobility_model = model;
      const auto scaling   = CahnHilliard::get_adaptive_mobility_scaling(param);
      auto       velocity  = u;
      auto       gradient  = g;
      if (state == 1)
        velocity = 0.;
      if (state == 2)
        gradient = 0.;
      if (state == 3)
      {
        velocity    = 0.;
        velocity[0] = -g[1];
        velocity[1] = g[0];
      }
      const auto evaluate =
        CahnHilliard::get_mobility_evaluation_function(param);
      const auto base = evaluate(param, 0.3, velocity, gradient, scaling);
      const auto hess = CahnHilliard::evaluate_adaptive_mobility_hessians(
        param, velocity, gradient, scaling);
      const double h = 2e-6;
      for (const bool perturb_velocity : {false, true})
      {
        const auto plus =
          evaluate(param,
                   0.3,
                   velocity + (perturb_velocity ? h : 0.) * direction,
                   gradient + (perturb_velocity ? 0. : h) * direction,
                   scaling);
        const auto minus =
          evaluate(param,
                   0.3,
                   velocity - (perturb_velocity ? h : 0.) * direction,
                   gradient - (perturb_velocity ? 0. : h) * direction,
                   scaling);
        const double derivative =
          (perturb_velocity ? base.derivative_wrt_velocity :
                              base.derivative_wrt_tracer_gradient) *
          direction;
        AssertThrow(
          std::abs((plus.value - minus.value) / (2. * h) - derivative) < 2e-8,
          ExcMessage(
            "Mobility first derivative differs from finite differences."));
        const auto velocity_hessian =
          perturb_velocity ? hess.velocity_velocity : hess.velocity_gradient;
        const auto gradient_hessian = perturb_velocity ?
                                        transpose(hess.velocity_gradient) :
                                        hess.gradient_gradient;
        AssertThrow(
          ((plus.derivative_wrt_velocity - minus.derivative_wrt_velocity) /
             (2. * h) -
           velocity_hessian * direction)
              .norm() < 2e-8,
          ExcMessage(
            "Mobility velocity Hessian differs from finite differences."));
        AssertThrow(
          ((plus.derivative_wrt_tracer_gradient -
            minus.derivative_wrt_tracer_gradient) /
             (2. * h) -
           gradient_hessian * direction)
              .norm() < 2e-8,
          ExcMessage(
            "Mobility gradient Hessian differs from finite differences."));
      }
      if (state == 1)
      {
        const double expected = model == Model::adaptive_normal_velocity ?
                                  scaling.delta :
                                  scaling.coefficient * scaling.delta;
        AssertThrow(std::abs(base.value - expected) < 1e-14,
                    ExcMessage("Incorrect mobility at rest."));
      }
    }
}

void check_adaptive_values()
{
  Parameters::CahnHilliard<2> param;
  using Model                   = Parameters::CahnHilliard<2>::MobilityModel;
  param.surface_tension         = 2. * std::sqrt(2.) / 3.;
  param.epsilon_interface       = 1.;
  param.mobility_scaling_factor = 1. / std::sqrt(2.);
  param.mobility_regularization = 8.;
  param.velocity_regularization = 4.;
  Tensor<1, 2> u, g;
  u[0]                 = 3.;
  g[0]                 = 2.;
  param.mobility_model = Model::adaptive_normal_velocity;
  auto evaluate        = CahnHilliard::get_mobility_evaluation_function(param);
  auto result          = evaluate(
    param, 0., u, g, CahnHilliard::get_adaptive_mobility_scaling(param));
  // sigma_tilde=1 and A=1 give sqrt((3*2)^2+8^2)=10.
  AssertThrow(std::abs(result.value - 10.) < 1e-13 &&
                std::abs(result.derivative_wrt_velocity[0] - 1.2) < 1e-13,
              ExcMessage("Incorrect normal-velocity mobility scaling."));
  param.mobility_model          = Model::adaptive_velocity;
  param.mobility_scaling_factor = 2.;
  evaluate = CahnHilliard::get_mobility_evaluation_function(param);
  result   = evaluate(
    param, 0., u, g, CahnHilliard::get_adaptive_mobility_scaling(param));
  // A=2 gives 2*sqrt(3^2+4^2)=10.
  AssertThrow(std::abs(result.value - 10.) < 1e-13 &&
                std::abs(result.derivative_wrt_velocity[0] - 1.2) < 1e-13,
              ExcMessage("Incorrect velocity mobility scaling."));
}

void check_degenerate()
{
  Parameters::CahnHilliard<2> param;
  ParameterHandler            prm;
  param.declare_parameters(prm);
  prm.parse_input_from_string(R"(
subsection Cahn Hilliard
  subsection mobility
    set model = degenerate
    set Function expression = 2*(1-x*x)^2
  end
end
)");
  param.read_parameters(prm);
  const auto evaluate = CahnHilliard::get_mobility_evaluation_function(param);
  const Tensor<1, 2> zero;
  auto               result = evaluate(param, 0.5, zero, zero, {});
  AssertThrow(std::abs(result.value - 1.125) < 1e-13 &&
                std::abs(result.derivative_wrt_tracer + 3.) < 1e-13 &&
                std::abs(result.second_derivative_wrt_tracer + 2.) < 1e-13,
              ExcMessage("Incorrect polynomial mobility or derivatives."));
  AssertThrow(evaluate(param, 1., zero, zero, {}).value == 0.,
              ExcMessage("Degenerate mobility must vanish in the pure phase."));
  param.mobility_tracer_limiter = true;
  for (const double phi : {-1.2, 1.2})
  {
    result = evaluate(param, phi, zero, zero, {});
    AssertThrow(result.value == 0. && result.derivative_wrt_tracer == 0. &&
                  result.second_derivative_wrt_tracer == 0.,
                ExcMessage("Mobility clamp chain rule is incorrect."));
  }
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  check_adaptive_values();
  check_degenerate();
  check_derivatives<2>();
  check_derivatives<3>();
  deallog << "Mobility laws: values, derivatives, clamp and Hessians in "
             "2D/3D OK"
          << std::endl;
}
