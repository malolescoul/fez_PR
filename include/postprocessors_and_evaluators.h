#ifndef POSTPROCESSORS_AND_EVALUATORS_H
#define POSTPROCESSORS_AND_EVALUATORS_H

#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <field_postprocessors.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <types.h>
#include <utilities.h>

#include <type_traits>

namespace PostProcessingTools
{
  using namespace dealii;

  /**
   * A DataPostProcessor to compute the vorticity. No particular treatment is
   * applied to the vorticity field, thus if the velocity is continuous, the
   * output vorticity will be discontinuous.
   */
  template <int dim>
  class VorticityPostProcessor : public DataPostprocessorVector<dim>
  {
  public:
    /**
     * Constructor.
     */
    VorticityPostProcessor(const ComponentOrdering &ordering);

    /**
     * Evaluate the vorticity as curl(u).
     */
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override;

  private:
    const unsigned int u_lower;
  };

  /**
   * An Evaluator to compute the vorticity at quadrature nodes, store some
   * data needed to perform these computations (e.g., required update flags in
   * the main solver's FEValues, etc.), and specify the types involved.
   *
   * This class thus does a similar job as VorticityPostProcessor above, but
   * with a few more data due to the fact that we're computing and outputting
   * the data ourselves, instead of using the DataPostprocessor facilities.
   */
  template <int dim>
  class VorticityEvaluator
  {
  public:
    /**
     * Some bookkeeping data.
     *
     * The vorticity (curl of velocity) is a scalar field in 2D, and a vector
     * field in 3D. In deal.II, this is represented by a curl_type, which is
     * always a Tensor<1, dim> in 3D, but in 2D, depending on the version of the
     * library, it can be either a Tensor<1, 1> or a double. The former needs to
     * be indexed [0] to return its value. The shape functions, on the other
     * hand, are always double for scalars.
     */
    static constexpr unsigned int n_components = dim == 2 ? 1 : 3;
    using curl_type =
      typename FEValuesViews::Vector<dim>::template solution_curl_type<double>;
    using quantity_type = curl_type;
    using shape_type    = std::conditional_t<dim == 2, double, Tensor<1, dim>>;

    /**
     * Constructor.
     */
    VorticityEvaluator(const ComponentOrdering    &ordering,
                       const ParameterReader<dim> &param);

    /**
     * Reinit member data. This evaluator does not require any, so this does
     * nothing.
     */
    void reinit(const PostprocessorAtDofBase<dim> &) {}

    /**
     * Return the name of each of the postprocessed component.
     */
    std::vector<std::string> get_data_names() const;

    /**
     * Return the interpretation of each postprocessed component.
     */
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const;

    /**
     * Return the update flags required by this evaluator to compute the
     * vorticity. These flags are given to the FEValues of the main solver.
     */
    UpdateFlags get_main_solver_update_flags() const;

    /**
     * Evaluate the vorticity.
     */
    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(const VectorType           &solution,
                           const FEValuesType         &fe_values,
                           std::vector<quantity_type> &computed_values) const
    {
      fe_values[velocity_extractor].get_function_curls(solution,
                                                       computed_values);
    }

    /**
     * Return the @p index-th entry of @p values as a shape_type object.
     * This function basically exists to return a Tensor<1, 1> as a double,
     * as it's the type used for the 2D curl until deal.II version 9.7.
     * It's likely this function can be removed at some point.
     */
    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      // Starting with deal.II v9.8, curl_type is a scalar in 2D, but in v9.7 it
      // is a Tensor<1, 1>. The return type of this function will be multiplied
      // by shape functions to yield a double, so here we simply return a
      // shape_type. If both types are the same, simply return the object,
      // otherwise return the value of the Tensor<1, 1> object in 2D.
      if constexpr (std::is_same_v<curl_type, shape_type>)
        return values[index];
      else
      {
        static_assert(dim == 2);
        return values[index][0];
      }
    }

  public:
    /**
     * The parameters associated with the vorticity.
     */
    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    /**
     * Extractor for the velocity in the main solver's FEValues.
     */
    FEValuesExtractors::Vector velocity_extractor;
  };

  /**
   * A DataPostProcessor to compute the Q criterion.
   */
  template <int dim>
  class QCriterionPostProcessor : public DataPostprocessorScalar<dim>
  {
  public:
    /**
     * Constructor.
     */
    QCriterionPostProcessor(const ComponentOrdering &ordering);

    /**
     * Evaluate the Q-criterion from the velocity gradient.
     */
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override;

  private:
    const unsigned int u_lower;
  };

  /**
   * An Evaluator to compute the Q-criterion at quadrature nodes, store some
   * data needed to perform these computations (e.g., required update flags in
   * the main solver's FEValues, etc.), and specify the types involved.
   */
  template <int dim>
  class QCriterionEvaluator
  {
  public:
    static constexpr unsigned int n_components = 1;
    using quantity_type                        = double;
    using shape_type                           = double;

    /**
     * Constructor.
     */
    QCriterionEvaluator(const ComponentOrdering    &ordering,
                        const ParameterReader<dim> &param);

    /**
     * Resize velocity_gradients to the number of quadrature points.
     */
    void reinit(const PostprocessorAtDofBase<dim> &projection_base);

    /**
     * Return the name of the postprocessed component.
     */
    std::vector<std::string> get_data_names() const;

    /**
     * Return the interpretation of the postprocessed component.
     */
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const;

    /**
     * Return the update flags required by this evaluator to compute the
     * Q-criterion. These flags are given to the FEValues of the main solver.
     */
    UpdateFlags get_main_solver_update_flags() const;

    /**
     * Evaluate the Q-criterion.
     */
    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(const VectorType           &solution,
                           const FEValuesType         &fe_values,
                           std::vector<quantity_type> &computed_values)
    {
      AssertDimension(computed_values.size(), velocity_gradients.size());
      fe_values[velocity_extractor].get_function_gradients(solution,
                                                           velocity_gradients);
      const unsigned int size = computed_values.size();
      for (unsigned int i = 0; i < size; ++i)
        computed_values[i] = compute_q_criterion<dim>(velocity_gradients[i]);
    }

    /**
     * Return the @p index-th entry of @p values as a shape_type object.
     * This function is trivial here, and it's likely that it can be removed at
     * some point.
     */
    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

  public:
    /**
     * The parameters associated with the Q-criterion.
     */
    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    /**
     * Extractor for the velocity in the main solver's FEValues.
     */
    FEValuesExtractors::Vector velocity_extractor;

    /**
     * Used to get the velocity gradients at the quadrature nodes, then compute
     * the criterion from them.
     */
    std::vector<Tensor<2, dim>> velocity_gradients;
  };

  /**
   * Reconstruct the Abels pressure p + phi*mu at visualization points.
   * Use the solved phase directly, as in the capillary term phi*grad(mu).
   */
  template <int dim>
  class AbelsPressurePostProcessor : public DataPostprocessorScalar<dim>
  {
  public:
    AbelsPressurePostProcessor(const ComponentOrdering &ordering)
      : DataPostprocessorScalar<dim>("pressure_abels", update_values)
      , p_lower(ordering.p_lower)
      , phi_lower(ordering.phi_lower)
      , mu_lower(ordering.mu_lower)
    {}

    void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_values.size(),
                      computed_quantities.size());
      for (unsigned int q = 0; q < computed_quantities.size(); ++q)
      {
        const auto &values = input_data.solution_values[q];
        AssertIndexRange(p_lower, values.size());
        AssertIndexRange(phi_lower, values.size());
        AssertIndexRange(mu_lower, values.size());
        AssertDimension(computed_quantities[q].size(), 1);
        computed_quantities[q][0] =
          values[p_lower] + values[phi_lower] * values[mu_lower];
      }
    }

  private:
    const unsigned int p_lower;
    const unsigned int phi_lower;
    const unsigned int mu_lower;
  };

  /** Reconstruct the CHNS mixture density at quadrature points. */
  template <int dim>
  class DensityEvaluator
  {
  public:
    static constexpr unsigned int n_components = 1;
    using quantity_type                        = double;
    using shape_type                           = double;

    DensityEvaluator(const ComponentOrdering    &ordering,
                     const ParameterReader<dim> &parameters)
      : param(parameters.postprocessing.density)
      , tracer_extractor(ordering.phi_lower)
      , tracer_limiter(
          CahnHilliard::get_limiter_function(parameters.cahn_hilliard))
    {
      AssertThrow(ordering.phi_lower != numbers::invalid_unsigned_int,
                  ExcMessage("Density postprocessing requires a CHNS tracer."));
      AssertThrow(parameters.physical_properties.fluids.size() >= 2,
                  ExcMessage("Density postprocessing requires two fluids."));
      density_0 = parameters.physical_properties.fluids[0].density;
      density_1 = parameters.physical_properties.fluids[1].density;
    }

    void reinit(const PostprocessorAtDofBase<dim> &projection_base)
    {
      tracer_values.resize(projection_base.get_n_q_points());
    }

    std::vector<std::string> get_data_names() const { return {"density"}; }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const
    {
      return {DataComponentInterpretation::component_is_scalar};
    }

    UpdateFlags get_main_solver_update_flags() const { return update_values; }

    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(const VectorType           &solution,
                           const FEValuesType         &fe_values,
                           std::vector<quantity_type> &computed_values)
    {
      AssertDimension(computed_values.size(), tracer_values.size());
      fe_values[tracer_extractor].get_function_values(solution, tracer_values);
      for (unsigned int q = 0; q < computed_values.size(); ++q)
        computed_values[q] = evaluate_density(tracer_values[q]);
    }

    /** Same material law for quadrature and direct visualization output. */
    double evaluate_density(const double tracer) const
    {
      const double marker = tracer_limiter(tracer);
      return CahnHilliard::linear_mixing(marker, density_0, density_1);
    }

    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    FEValuesExtractors::Scalar          tracer_extractor;
    double                              density_0;
    double                              density_1;
    CahnHilliard::TracerLimiterFunction tracer_limiter;
    std::vector<double>                 tracer_values;
  };

  /** Evaluate density directly from the tracer at visualization points. */
  template <int dim>
  class DensityPostProcessor : public DataPostprocessorScalar<dim>
  {
  public:
    DensityPostProcessor(const ComponentOrdering    &ordering,
                         const ParameterReader<dim> &parameters)
      : DataPostprocessorScalar<dim>("density", update_values)
      , phi_lower(ordering.phi_lower)
      , evaluator(ordering, parameters)
    {}

    void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_values.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < computed_quantities.size(); ++p)
      {
        AssertIndexRange(phi_lower, input_data.solution_values[p].size());
        AssertDimension(computed_quantities[p].size(), 1);
        computed_quantities[p][0] =
          evaluator.evaluate_density(input_data.solution_values[p][phi_lower]);
      }
    }

  private:
    const unsigned int          phi_lower;
    const DensityEvaluator<dim> evaluator;
  };

  /** Evaluate mobility on the physical fields, using the same law as assembly.
   */
  template <int dim>
  class MobilityEvaluator
  {
  public:
    static constexpr unsigned int n_components = 1;
    using quantity_type                        = double;
    using shape_type                           = double;

    MobilityEvaluator(const ComponentOrdering    &ordering,
                      const ParameterReader<dim> &parameters)
      : param(parameters.postprocessing.mobility)
      , cahn_hilliard(parameters.cahn_hilliard)
      , velocity_extractor(ordering.u_lower)
      , tracer_extractor(ordering.phi_lower)
      , adaptive(CahnHilliard::is_adaptive_mobility_model(cahn_hilliard))
      , degenerate(cahn_hilliard.mobility_model ==
                   Parameters::CahnHilliard<dim>::MobilityModel::degenerate)
      , gradient_dependent(cahn_hilliard.mobility_model ==
                           Parameters::CahnHilliard<
                             dim>::MobilityModel::adaptive_normal_velocity)
      , mobility_function(
          CahnHilliard::get_mobility_evaluation_function(cahn_hilliard))
    {
      AssertThrow(
        ordering.u_lower != numbers::invalid_unsigned_int &&
          ordering.phi_lower != numbers::invalid_unsigned_int,
        ExcMessage(
          "Mobility postprocessing requires CHNS velocity and tracer."));
    }

    void reinit(const PostprocessorAtDofBase<dim> &projection_base)
    {
      if (degenerate)
        tracer_values.resize(projection_base.get_n_q_points());
      if (adaptive)
        velocity_values.resize(projection_base.get_n_q_points());
      if (gradient_dependent)
        tracer_gradients.resize(projection_base.get_n_q_points());
    }

    std::vector<std::string> get_data_names() const { return {"mobility"}; }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const
    {
      return {DataComponentInterpretation::component_is_scalar};
    }

    UpdateFlags get_main_solver_update_flags() const
    {
      return gradient_dependent ? update_values | update_gradients :
                                  update_values;
    }

    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(const VectorType           &solution,
                           const FEValuesType         &fe_values,
                           std::vector<quantity_type> &computed_values)
    {
      if (!adaptive && !degenerate)
      {
        std::fill(computed_values.begin(),
                  computed_values.end(),
                  cahn_hilliard.mobility);
        return;
      }
      if (adaptive)
        fe_values[velocity_extractor].get_function_values(solution,
                                                          velocity_values);
      if (degenerate)
        fe_values[tracer_extractor].get_function_values(solution,
                                                        tracer_values);
      if (gradient_dependent)
        fe_values[tracer_extractor].get_function_gradients(solution,
                                                           tracer_gradients);
      const auto scaling =
        CahnHilliard::get_adaptive_mobility_scaling(cahn_hilliard);
      for (unsigned int q = 0; q < computed_values.size(); ++q)
        computed_values[q] =
          mobility_function(cahn_hilliard,
                            degenerate ? tracer_values[q] : 0.,
                            adaptive ? velocity_values[q] : Tensor<1, dim>(),
                            gradient_dependent ? tracer_gradients[q] :
                                                 Tensor<1, dim>(),
                            scaling)
            .value;
    }

    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    const Parameters::CahnHilliard<dim>          &cahn_hilliard;
    FEValuesExtractors::Vector                    velocity_extractor;
    FEValuesExtractors::Scalar                    tracer_extractor;
    const bool                                    adaptive;
    const bool                                    degenerate;
    std::vector<double>                           tracer_values;
    const bool                                    gradient_dependent;
    CahnHilliard::MobilityEvaluationFunction<dim> mobility_function;
    std::vector<Tensor<1, dim>>                   velocity_values;
    std::vector<Tensor<1, dim>>                   tracer_gradients;
  };

  /**
   * Compute mesh velocity, for solvers solving for the mesh position.
   * The resulting field uses the same finite element representation as the mesh
   * position field in the main solver, so that we can simply take the time
   * derivative of the nodal values. Consequently, the "degree" parameter is
   * ignored for this class.
   */
  template <int dim>
  class MeshVelocityPostprocessor : public PostprocessorAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     *
     * FIXME: the mapping and cell_quadrature are unused, since this
     * postprocessor does not need an FEValues. The PostprocessorAtDofBase could
     * be simplified to create an FEValues only if needed.
     */
    MeshVelocityPostprocessor(const ComponentOrdering    &ordering,
                              const ParameterReader<dim> &param,
                              const Mapping<dim>         &mapping,
                              const DoFHandler<dim>      &solver_dof_handler,
                              const Quadrature<dim>      &cell_quadrature);

  protected:
    /**
     * Take the time derivative at dofs of the mesh position solution.
     */
    void
    do_postprocess(const LA::ParVectorType              &present_solution,
                   const std::vector<LA::ParVectorType> &previous_solutions,
                   const TimeHandler                    &time_handler) override;

  private:
    /**
     * Owned mesh position degrees of freedom in the main solver's dof_handler.
     */
    IndexSet solver_owned_position_dofs;
  };
} // namespace PostProcessingTools

#endif
