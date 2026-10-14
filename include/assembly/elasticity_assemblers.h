#ifndef ELASTICITY_ASSEMBLERS_H
#define ELASTICITY_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <components_ordering.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <scratch_data.h>

#include <algorithm>
#include <cmath>

template <int dim>
class ScratchDataElasticity;

namespace Assembly
{
  namespace Elasticity
  {
    /**
     * Regularized compression coefficient of the Cahn-Hilliard moving-mesh
     * forcing. The raw coefficient phi / (1 - gamma^2 phi^2) is singular as
     * gamma*phi -> 1, so phi is saturated through a tanh before being used.
     * Both the value and its derivative w.r.t. the phase are returned (the
     * derivative is used to linearize the forcing).
     */
    struct MeshForcingFactor
    {
      double value;
      double derivative;
    };

    inline MeshForcingFactor mesh_forcing_factor(const double phase_value,
                                                 const double gamma)
    {
      constexpr double phi_max_user = 0.998;
      const double     phi_max_safe =
        std::min(phi_max_user, 0.98 / std::max(gamma, 1e-14));

      const double z                    = phase_value / phi_max_safe;
      const double tanh_z               = std::tanh(z);
      const double regularized_phase    = phi_max_safe * tanh_z;
      const double regularized_jacobian = 1. - tanh_z * tanh_z;
      const double denominator =
        1. - gamma * gamma * regularized_phase * regularized_phase;
      const double support            = 1. / denominator;
      const double support_derivative = 2. * gamma * gamma * regularized_phase *
                                        regularized_jacobian /
                                        (denominator * denominator);

      return {phase_value * support,
              support + phase_value * support_derivative};
    }

    /**
     * Smooth odd approximation of sign(psi) * |psi|^q used to equalize the
     * enlarged mesh-forcing marker before it enters the regularized factor. The
     * small delta keeps the map differentiable at psi=0 for Newton. With q = 1
     * the map is the identity (no-op). For 0 < q < 1 weak marker values are
     * amplified (the mesh reacts earlier across the enlarged interface); q > 1
     * damps them. Both the value and its derivative are returned.
     */
    inline MeshForcingFactor
    smooth_power_equalized_phase(const double phase_value,
                                 const double exponent)
    {
      if (std::abs(exponent - 1.) < 1e-14)
        return {phase_value, 1.};

      constexpr double delta = 2e-2;
      const double     a     = phase_value * phase_value + delta * delta;

      return {phase_value * std::pow(a, 0.5 * (exponent - 1.)),
              std::pow(a, 0.5 * (exponent - 3.)) *
                (delta * delta + exponent * phase_value * phase_value)};
    }

    /**
     * Peak of the dimensionless enlarged-compression lobe associated with the
     * one-dimensional Helmholtz marker.  Away from the much thinner physical
     * interface, psi - L^2 psi'' = sign(x) gives
     *
     *   L |grad psi| = 1 - |psi|.
     *
     * Hence the lobe shape, with s = |psi| in [0,1], is
     *
     *   (1-s) |factor(equalize_q(s))|.
     *
     * A short, deterministic one-dimensional search is sufficient here.  Its
     * result is stored by the forcing assembler and reused in all FE loops.
     */
    inline double enlarged_mesh_forcing_lobe_peak(const double gamma,
                                                  const double exponent)
    {
      const auto lobe = [gamma, exponent](const double s) {
        const double equalized =
          smooth_power_equalized_phase(s, exponent).value;
        return (1. - s) * std::abs(mesh_forcing_factor(equalized, gamma).value);
      };

      // First bracket the global maximum.  For positive q the lobe is smooth
      // and unimodal; the coarse scan also keeps the search robust near very
      // small exponents and strong regularization.
      constexpr unsigned int n_intervals = 128;
      unsigned int           best_index  = 0;
      double                 best_value  = 0.;
      for (unsigned int i = 0; i <= n_intervals; ++i)
      {
        const double value = lobe(static_cast<double>(i) / n_intervals);
        if (value > best_value)
        {
          best_value = value;
          best_index = i;
        }
      }

      double left =
        static_cast<double>(best_index == 0 ? 0 : best_index - 1) / n_intervals;
      double right =
        static_cast<double>(std::min(n_intervals, best_index + 1)) /
        n_intervals;

      constexpr double golden_ratio_conjugate = 0.6180339887498948482;
      double           x_left = right - golden_ratio_conjugate * (right - left);
      double           x_right = left + golden_ratio_conjugate * (right - left);
      double           value_left  = lobe(x_left);
      double           value_right = lobe(x_right);
      for (unsigned int iteration = 0; iteration < 48; ++iteration)
      {
        if (value_left < value_right)
        {
          left        = x_left;
          x_left      = x_right;
          value_left  = value_right;
          x_right     = left + golden_ratio_conjugate * (right - left);
          value_right = lobe(x_right);
        }
        else
        {
          right       = x_right;
          x_right     = x_left;
          value_right = value_left;
          x_left      = right - golden_ratio_conjugate * (right - left);
          value_left  = lobe(x_left);
        }
      }

      return std::max(best_value, std::max(value_left, value_right));
    }

    /**
     * Automatic amplitude normalization of the equalized enlarged forcing.
     * The q=1 lobe is the reference, so changing q primarily moves/reshapes the
     * lobes without changing their theoretical peak compression.
     */
    inline double enlarged_mesh_forcing_normalization(const double gamma,
                                                      const double exponent)
    {
      if (std::abs(exponent - 1.) < 1e-14)
        return 1.;

      const double reference_peak = enlarged_mesh_forcing_lobe_peak(gamma, 1.);
      const double equalized_peak =
        enlarged_mesh_forcing_lobe_peak(gamma, exponent);

      return equalized_peak > 1e-14 ? reference_peak / equalized_peak : 1.;
    }

    /**
     * Regularized compression coefficient for the enlarged marker psi: the
     * equalization map is chained into mesh_forcing_factor, so the returned
     * value is factor(equalize(psi)) and the derivative is the product of the
     * two chain-rule terms (w.r.t. psi).
     */
    inline MeshForcingFactor
    enlarged_mesh_forcing_factor(const double phase_value,
                                 const double gamma,
                                 const double exponent,
                                 const double normalization)
    {
      const MeshForcingFactor equalized =
        smooth_power_equalized_phase(phase_value, exponent);
      MeshForcingFactor factor = mesh_forcing_factor(equalized.value, gamma);
      factor.value *= normalization;
      factor.derivative *= equalized.derivative * normalization;
      return factor;
    }

    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim, typename ScratchData, typename CopyData>
    void setup_assemblers(
      const ParameterReader<dim> &param,
      const ComponentOrdering    &ordering,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers);

    /**
     *
     */
    template <int dim, typename ScratchData, typename CopyData>
    class LinearElasticityAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      LinearElasticityAssembler(const ParameterReader<dim> &param,
                                const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };

    /**
     *
     */
    template <int dim, typename ScratchData, typename CopyData>
    class NeoHookeanAssembler : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      NeoHookeanAssembler(const ParameterReader<dim> &param,
                          const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };

    /**
     *
     */
    template <int dim, typename ScratchData, typename CopyData>
    class OgdenHyperelasticityAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      OgdenHyperelasticityAssembler(const ParameterReader<dim> &param,
                                    const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };

    /**
     * Assemble the extra matrix contribution when the source term depends on
     * the current mesh position f(x(X)).
     */
    template <int dim, typename ScratchData, typename CopyData>
    class CurrentMeshSourceAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      CurrentMeshSourceAssembler(const ParameterReader<dim> &param,
                                 const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * This assembler has no additional rhs to assemble, as the source term
       * is already assembled in the main elasticity assembler.
       */
      virtual void assemble_rhs(const ScratchData &, CopyData &) const override
      {}

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };

    /**
     * Elasticity source term depending on the Cahn-Hilliard Navier-Stokes phase
     * marker and velocity: compression * epsilon * factor(phi) * grad(phi)
     * minus transport * epsilon^2 * (u_ALE . grad(phi)) * grad(phi).
     * A positive compression factor concentrates the mesh near the interface.
     *
     * The elasticity presolver uses an analytic phase and its Hessian. The
     * moving CHNS solver uses the discrete phase and differentiates its
     * gradient with respect to the mesh position, phase and velocity.
     * In the enlarged solver, psi adds a second compression term and replaces
     * phi in the transport term, with length L instead of epsilon.
     * Instantiated for elasticity and CHNS scratch data with a moving mesh.
     */
    template <int dim, typename ScratchData, typename CopyData>
    class SourceFromCHNSTracerAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      SourceFromCHNSTracerAssembler(const ParameterReader<dim> &param,
                                    const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {
        if constexpr (!std::is_same_v<ScratchData, ScratchDataElasticity<dim>>)
          if constexpr (ScratchData::enable_enlarged)
            enlarged_normalization = enlarged_mesh_forcing_normalization(
              param.cahn_hilliard.mff_regularization_gamma,
              param.cahn_hilliard.mff_enlarged_lobe_position_exponent);
      }

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &, CopyData &) const override;

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;

    private:
      double enlarged_normalization = 1.;
    };
  } // namespace Elasticity
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace Elasticity
  {
    template <int dim, typename ScratchData, typename CopyData>
    void setup_assemblers(
      const ParameterReader<dim> &param,
      const ComponentOrdering    &ordering,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers)
    {
      const auto &solid = param.physical_properties.pseudosolids[0];

      switch (solid.constitutive_model)
      {
        case Parameters::PseudoSolid<dim>::ConstitutiveModel::linear_elasticity:
          assemblers.emplace_back(
            std::make_unique<
              LinearElasticityAssembler<dim, ScratchData, CopyData>>(param,
                                                                     ordering));
          break;
        case Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean:
          assemblers.emplace_back(
            std::make_unique<NeoHookeanAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
          break;
        case Parameters::PseudoSolid<dim>::ConstitutiveModel::ogden:
          assemblers.emplace_back(
            std::make_unique<
              OgdenHyperelasticityAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
          break;
        default:
          DEAL_II_ASSERT_UNREACHABLE();
      }

      // Custom (user-defined) source term on the current mesh.
      if (param.elasticity.enable_source_term_on_current_mesh)
        assemblers.emplace_back(
          std::make_unique<
            CurrentMeshSourceAssembler<dim, ScratchData, CopyData>>(param,
                                                                    ordering));

      // Cahn-Hilliard moving-mesh forcing ("chns form" path): function mode for
      // the elasticity presolver (analytic phi) and field mode for the full
      // CHNS-ALE solver (FE phi). Both share this assembler.
      const bool with_chns_form_forcing =
        param.cahn_hilliard.mff_source_term ==
        Parameters::CahnHilliard<dim>::MeshForcingSourceTerm::chns_form;

      if constexpr (std::is_same_v<ScratchData, ScratchDataElasticity<dim>> ||
                    std::is_same_v<
                      ScratchData,
                      NavierStokesScratch::ScratchDataCHNS<dim, true>> ||
                    std::is_same_v<
                      ScratchData,
                      NavierStokesScratch::ScratchDataCHNS<dim, true, true>>)
      {
        // Assemble this term only when its enable flag is set.
        if (with_chns_form_forcing)
          assemblers.emplace_back(
            std::make_unique<
              SourceFromCHNSTracerAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
      }
    }
  } // namespace Elasticity
} // namespace Assembly

#endif
