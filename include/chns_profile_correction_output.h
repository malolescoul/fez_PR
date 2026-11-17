#ifndef CHNS_PROFILE_CORRECTION_OUTPUT_H
#define CHNS_PROFILE_CORRECTION_OUTPUT_H

#include <chns_profile_coefficient.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_postprocessor.h>

namespace CahnHilliard
{
  /** Evaluate the same nodal inverse-profile reconstruction as assembly.
   * A postprocessor of the interpolated phase gradient would export a
   * different flux. Local FEValues also support volume and skin patches.
   * The solution and mapping must outlive build_patches().
   */
  template <int dim, typename VectorType>
  class ProfileCorrectionOutput : public dealii::DataPostprocessor<dim>
  {
  public:
    ProfileCorrectionOutput(
      const Parameters::CahnHilliard<dim>               &ch,
      const dealii::Mapping<dim>                        &mapping,
      const VectorType                                  &solution,
      const unsigned                                     phase_component,
      const unsigned                                     velocity_component,
      std::shared_ptr<const CHNSProfileGeometry::Fields> geometry,
      const double                                       constant_coefficient)
      : ch(ch)
      , mapping(mapping)
      , solution(solution)
      , phase(phase_component)
      , velocity(velocity_component)
      , geometry(std::move(geometry))
      , constant_coefficient(constant_coefficient)
    {}

    std::vector<std::string> get_names() const override
    {
      std::vector<std::string> names{"profile_correction_strength",
                                     "profile_correction_coefficient"};
      names.insert(names.end(), dim, "profile_correction_flux");
      if (ch.local_profile_correction())
      {
        names.push_back("profile_correction_geometric_penalty");
        names.push_back("profile_correction_retained_fraction");
      }
      if (ch.local_profile_correction())
        for (const auto name : {"profile_correction_feedback_gain",
                                "profile_correction_feedback_support",
                                "profile_correction_width_error",
                                "profile_correction_feedback_confidence",
                                "profile_correction_timescale_strength"})
          names.push_back(name);
      return names;
    }

    std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      using namespace dealii::DataComponentInterpretation;
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
        result(2, component_is_scalar);
      result.insert(result.end(), dim, component_is_part_of_vector);
      if (ch.local_profile_correction())
        result.insert(result.end(), 2, component_is_scalar);
      if (ch.local_profile_correction())
        result.insert(result.end(), 5, component_is_scalar);
      return result;
    }

    dealii::UpdateFlags get_needed_update_flags() const override
    {
      return dealii::update_quadrature_points;
    }

    void evaluate_vector_field(
      const dealii::DataPostprocessorInputs::Vector<dim> &input,
      std::vector<dealii::Vector<double>>                &out) const override
    {
      using namespace dealii;
      if (!ch.enable_profile_correction)
      {
        for (auto &entry : out)
          entry = 0.;
        return;
      }
      const auto              cell = input.template get_cell<dim>();
      std::vector<Point<dim>> points;
      for (const auto &p : input.evaluation_points)
        points.push_back(mapping.transform_real_to_unit_cell(cell, p));
      FEValues<dim> values(mapping,
                           cell->get_fe(),
                           Quadrature<dim>(points),
                           update_values | update_gradients | update_hessians |
                             update_inverse_jacobians);
      values.reinit(cell);
      Vector<double> nodal(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_values(solution, nodal);
      for (unsigned j = 0; j < nodal.size(); ++j)
        nodal[j] =
          cell->get_fe().system_to_component_index(j).first == phase.component ?
            inverse_profile(nodal[j]).value :
            0.;
      std::vector<Tensor<2, dim>> grad_u(points.size());
      values[velocity].get_function_gradients(solution, grad_u);
      const double eps = ch.epsilon_interface;
      const double base =
        ch.mobility * (3. / (2. * std::sqrt(2.))) * ch.surface_tension / eps;
      for (unsigned q = 0; q < points.size(); ++q)
      {
        ReconstructedProfile<dim> profile;
        double                    inverse_value = 0.;
        for (unsigned j = 0; j < nodal.size(); ++j)
        {
          inverse_value += nodal[j] * values[phase].value(j, q);
          profile.inverse_gradient += nodal[j] * values[phase].gradient(j, q);
          profile.inverse_hessian += nodal[j] * values[phase].hessian(j, q);
        }
        profile.value  = std::tanh(inverse_value);
        auto geometric = geometry ?
                           geometry->evaluate(input.evaluation_points[q]) :
                           CHNSProfileOverlap::PenaltyValue{};
        if constexpr (dim == 2)
          if (geometry && !geometry->reference_cells.empty())
            geometric = geometry->evaluate_on_cell(cell->id().to_string(),
                                                   {points[q][0], points[q][1]},
                                                   values.inverse_jacobian(q));
        LocalProfileCoefficient<dim> coefficient(constant_coefficient);
        if (ch.local_profile_correction())
          coefficient =
            LocalProfileCoefficient<dim>(profile.inverse_gradient,
                                         profile.inverse_hessian,
                                         grad_u[q],
                                         {},
                                         eps,
                                         base,
                                         ch.profile_correction_timescale_factor,
                                         geometric);
        const auto flux = coefficient.apply(evaluate_profile_correction(
          profile.value, profile.gradient(), Tensor<2, dim>(), eps, false));
        out[q][0]       = ch.local_profile_correction() ?
                            coefficient.value / (2. * base) :
                            ch.profile_correction_strength;
        out[q][1]       = coefficient.value;
        for (unsigned d = 0; d < dim; ++d)
          out[q][2 + d] = flux.flux[d];
        if (ch.local_profile_correction())
        {
          out[q][2 + dim] = geometric.value;
          out[q][3 + dim] =
            coefficient.unattenuated_value() > 0. ?
              coefficient.value / coefficient.unattenuated_value() :
              1.;
        }
        if (ch.local_profile_correction())
        {
          const auto f    = geometric.feedback;
          out[q][4 + dim] = f.gain;
          out[q][5 + dim] = f.support;
          out[q][6 + dim] = f.error;
          out[q][7 + dim] = f.confidence;
          out[q][8 + dim] = coefficient.timescale_value() / (2. * base);
        }
      }
    }

  private:
    const Parameters::CahnHilliard<dim>                     &ch;
    const dealii::Mapping<dim>                              &mapping;
    const VectorType                                        &solution;
    const dealii::FEValuesExtractors::Scalar                 phase;
    const dealii::FEValuesExtractors::Vector                 velocity;
    const std::shared_ptr<const CHNSProfileGeometry::Fields> geometry;
    const double constant_coefficient;
  };
} // namespace CahnHilliard
#endif
