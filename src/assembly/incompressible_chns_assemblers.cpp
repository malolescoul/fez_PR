
#include <assembly/incompressible_chns_assemblers.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <parameter_reader.h>
#include <scratch_data.h>

namespace Assembly
{
  namespace IncompressibleCHNS
  {
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    void
    VolumeAssembler<dim, ScratchData, CopyData, assembly_flags>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd = scratch_data;

      if constexpr (BaseType::with_stabilization)
        Assert(
          sd.enable_stabilization,
          ExcMessage(
            "The assemblers for the incompressible Cahn-Hilliard Navier-Stokes "
            "equations are "
            "set to assemble SUPG-PSPG stabilization terms, but computation of "
            "the required data was not enabled in the provided ScratchData."));
      if constexpr (BaseType::with_tracer_stabilization)
        Assert(
          sd.enable_tracer_stabilization,
          ExcMessage(
            "The assemblers for the incompressible Cahn-Hilliard Navier-Stokes "
            "equations are "
            "set to assemble SUPG stabilization term for the tracer equation, "
            "but computation of "
            "the required data was not enabled in the provided ScratchData."));

      auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

      const double sigma_tilde_over_eps  = sd.sigma_tilde / sd.epsilon;
      const double sigma_tilde_times_eps = sd.sigma_tilde * sd.epsilon;
      const auto  &body_force            = sd.body_force;

      Tensor<1, dim> strong_residual_momentum;
      double         strong_residual_tracer;
      double         tau, tau_tracer;

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_moving = sd.JxW_moving[q];
        const double rho        = sd.density[q];
        const double eta        = sd.dynamic_viscosity[q];

        const auto  &dudt       = sd.present_velocity_time_derivatives[q];
        const auto  &u          = sd.present_velocity_values[q];
        const auto  &grad_u     = sd.present_velocity_gradients[q];
        const auto  &sym_grad_u = sd.present_velocity_sym_gradients[q];
        const double div_u      = sd.present_velocity_divergence[q];
        const auto  &source_u   = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          // ALE contribution
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }

        const auto &p        = sd.present_pressure_values[q];
        const auto &source_p = sd.source_term_pressure[q];

        const auto phase_flux = sd.get_phase_flux(q);
        const auto &diffusive_flux = sd.diffusive_flux[q];
        const auto &dphidt         = sd.tracer_time_derivatives[q];
        const auto &phi            = sd.tracer_values[q];
        const auto &grad_phi       = sd.tracer_gradients[q];
        const auto &mu             = sd.potential_values[q];
        const auto &grad_mu        = sd.potential_gradients[q];
        const auto &source_phi     = sd.source_term_tracer[q];
        const auto &source_mu      = sd.source_term_potential[q];

        auto supg_velocity = u_conv;
        if constexpr (BaseType::with_stabilization)
          supg_velocity +=
            (sd.diffusive_flux_factor / rho) * phase_flux;


        // Divergence of the mesh velocity, used by the conservative tracer
        // transport and the skew-symmetric momentum on moving meshes.
        double div_w = 0.;
        if constexpr (BaseType::with_moving_mesh)
          if (sd.conservative_tracer_transport || sd.skew_symmetric_momentum)
            div_w = trace(sd.present_mesh_velocity_gradients[q]);

        // Momentum: non-conservative form
        //   rho_i du/dt + rho (u_conv . grad) u + (J . grad) u,
        // or skew-symmetric form with the mass flux m = rho u_conv + J
        //   I_BDF + 1/2 (m . grad) u + 1/2 rho div(w) u,
        // tested with v, plus -1/2 ((m . grad) v) . u.
        Tensor<1, dim> to_mult_by_phi_u_i;
        Tensor<1, dim> mass_flux;
        if (sd.skew_symmetric_momentum)
        {
          mass_flux = rho * u_conv + sd.diffusive_flux_factor * phase_flux;
          to_mult_by_phi_u_i =
            sd.energy_time[q].inertia + 0.5 * (grad_u * mass_flux) +
            0.5 * rho * div_w * u - rho * body_force + phi * grad_mu + source_u;
        }
        else
          to_mult_by_phi_u_i = rho * (dudt + grad_u * u_conv - body_force) +
                               diffusive_flux + phi * grad_mu + source_u;

        // Tracer: advective form (u_conv . grad phi, psi), or conservative
        // form -(phi u_conv, grad psi) + (phi div(w), psi).
        const auto to_mult_by_phi_phi_i =
          sd.conservative_tracer_transport ?
            dphidt + phi * div_w + source_phi :
            dphidt + u_conv * grad_phi + source_phi;
        const auto to_mult_by_phi_mu_i =
          sd.skew_symmetric_momentum ?
            mu - sigma_tilde_over_eps * sd.energy_time[q].potential_derivative +
              source_mu :
            mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) + source_mu;

        const auto &phi_u          = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &phi_p          = sd.phi_p[q];
        const auto &grad_phi_p     = sd.grad_phi_p[q];
        const auto &phi_phi        = sd.shape_phi[q];
        const auto &grad_phi_phi   = sd.grad_shape_phi[q];
        const auto &phi_mu         = sd.shape_mu[q];
        const auto &grad_phi_mu    = sd.grad_shape_mu[q];

        double inv_rho = 0.;
        if constexpr (BaseType::with_stabilization)
        {
          tau                   = sd.tau_supg_velocity[q];
          inv_rho               = 1. / rho;

          // Compute strong residual of the momentum equation, in force units
          // (i.e. multiplied by the density). The SUPG and PSPG test operators
          // are ((u_conv + J/rho) . grad(v)) and grad(q) / rho.
          // Tau keeps its existing definition based on u_conv.
          strong_residual_momentum = CHNSEnergy::momentum_residual<dim>(sd, q);
        }

        if constexpr (BaseType::with_tracer_stabilization)
        {
          tau_tracer          = sd.tau_supg_tracer[q];

          // Compute strong residual of the tracer equation
          strong_residual_tracer =
            dphidt + u_conv * grad_phi - sd.get_phase_diffusion(q) +
            source_phi + (sd.conservative_tracer_transport ? phi * div_u : 0.);
        }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i   = sd.components[i];
          const bool         i_is_u   = this->ordering.is_velocity(comp_i);
          const bool         i_is_p   = this->ordering.is_pressure(comp_i);
          const bool         i_is_phi = this->ordering.is_tracer(comp_i);
          const bool         i_is_mu  = this->ordering.is_potential(comp_i);

          double local_rhs_i = i_is_p ? -phi_p[i] * (-div_u + source_p) : 0.;

          // Momentum equation
          if (i_is_u)
          {
            local_rhs_i -=
              phi_u[i] * to_mult_by_phi_u_i - div_phi_u[i] * p +
              2. * eta * scalar_product(sym_grad_phi_u[i], sym_grad_u);
            if (sd.skew_symmetric_momentum)
              local_rhs_i += 0.5 * u * (grad_phi_u[i] * mass_flux);

            if constexpr (BaseType::with_stabilization)
              // SUPG stabilization
              local_rhs_i -= tau * (strong_residual_momentum *
                                    (grad_phi_u[i] * supg_velocity));
          }

          // Continuity equation
          else if (i_is_p)
          {
            if constexpr (BaseType::with_stabilization)
              // PSPG stabilization
              local_rhs_i +=
                tau * inv_rho * (strong_residual_momentum * grad_phi_p[i]);
          }

          // Tracer equation
          else if (i_is_phi)
          {
            local_rhs_i -= phi_phi[i] * to_mult_by_phi_phi_i +
                           grad_phi_phi[i] * phase_flux;
            if (sd.conservative_tracer_transport)
              local_rhs_i += phi * (u_conv * grad_phi_phi[i]);

            if constexpr (BaseType::with_tracer_stabilization)
              // Tracer SUPG stabilization
              local_rhs_i -= tau_tracer * (u_conv * grad_phi_phi[i]) *
                             strong_residual_tracer;
          }

          // Potential equation
          else if (i_is_mu)
          {
            local_rhs_i -= phi_mu[i] * to_mult_by_phi_mu_i -
                           sigma_tilde_times_eps * (grad_phi_mu[i] * grad_phi);
          }

          local_rhs(i) += local_rhs_i * JxW_moving;
        }
      }

      if constexpr (BaseType::with_enlarged)
        assemble_psi_equation_rhs<dim>(this->ordering, sd, local_rhs);
    }

    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    void VolumeAssembler<dim, ScratchData, CopyData, assembly_flags>::
      assemble_matrix(const ScratchData &scratch_data,
                      CopyData          &copy_data) const
    {
      auto &sd           = scratch_data;
      auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

      const double bdf_c0                = sd.bdf_c0;
      const double sigma_tilde_over_eps  = sd.sigma_tilde / sd.epsilon;
      const double sigma_tilde_times_eps = sd.sigma_tilde * sd.epsilon;
      const auto  &body_force            = sd.body_force;

      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_momentum(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_potential(
        sd.dofs_per_cell);
      std::vector<double>         phi_u_j_x_grad_phi(sd.dofs_per_cell);
      std::vector<double>         to_mult_by_phi_phi_i(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> strong_residual_momentum_variation(
        sd.dofs_per_cell);
      std::vector<double> strong_residual_tracer_variation(sd.dofs_per_cell);
      Tensor<1, dim>      strong_residual_momentum;
      Tensor<1, dim>      strong_residual_momentum_variation_phi_phi;
      double              strong_residual_tracer;
      Tensor<1, dim>      supg_velocity_dot_grad_phi_u_i;
      Tensor<1, dim>      residual_dot_grad_phi_u_i;
      double              u_conv_dot_grad_phi_phi_i;
      Tensor<1, dim>      residual_tracer_dot_grad_phi_phi_i;
      double              tau, tau_tracer;

      std::vector<Tensor<1, dim>> supg_velocity_variation(
        BaseType::with_stabilization ? sd.dofs_per_cell : 0);
      std::vector<Tensor<1, dim>> supg_velocity_x_variation(
        BaseType::with_stabilization && BaseType::with_moving_mesh ?
          sd.dofs_per_cell :
          0);

      std::vector<double> mobility_variation(
        sd.variable_mobility ? sd.dofs_per_cell : 0);
      const bool additional_flux =
        sd.variable_mobility || !sd.profile_correction_flux.empty();
      std::vector<Tensor<1, dim>> additional_momentum_variation(
        additional_flux ? sd.dofs_per_cell : 0);
      std::vector<Tensor<1, dim>> correction_flux_variation(
        sd.profile_correction_flux.empty() ? 0 : sd.dofs_per_cell);

      // Skew-symmetric momentum: variation of the vector tested with v and of
      // the mass flux m = rho u_conv + J, for each trial function.
      const bool                  skew = sd.skew_symmetric_momentum;
      std::vector<Tensor<1, dim>> skew_momentum_variation(
        skew ? sd.dofs_per_cell : 0);
      std::vector<Tensor<1, dim>> mass_flux_variation(skew ? sd.dofs_per_cell :
                                                             0);
      // Mesh position variations of the skew-symmetric momentum: vector tested
      // with v, and vector w_j in -1/2 u . (grad(v) w_j).
      std::vector<Tensor<1, dim>> skew_momentum_x_variation(
        skew && BaseType::with_moving_mesh ? sd.dofs_per_cell : 0);
      std::vector<Tensor<1, dim>> skew_mass_flux_x_variation(
        skew && BaseType::with_moving_mesh ? sd.dofs_per_cell : 0);

      // Conservative tracer transport: variation of the terms tested with psi
      // and with grad(psi), for each trial function.
      const bool          conservative = sd.conservative_tracer_transport;
      std::vector<double> conservative_tracer_variation(
        conservative ? sd.dofs_per_cell : 0);
      std::vector<Tensor<1, dim>> conservative_tracer_grad_variation(
        conservative ? sd.dofs_per_cell : 0);

      const auto u_lower = this->ordering.u_lower;

      const SymmetricTensor<2, dim> identity_tensor =
        unit_symmetric_tensor<dim>();

      //
      // Moving mesh related data
      //
      const std::vector<Tensor<1, dim>> *phi_x;
      const std::vector<Tensor<2, dim>> *grad_phi_x_moving;
      const std::vector<Tensor<3, dim>> *hessian_phi_x_moving;
      // x-variation of the SUPG/PSPG strong residuals (frozen tau), used by the
      // momentum, continuity and tracer rows.
      std::vector<Tensor<1, dim>> strong_residual_momentum_x_variation(
        sd.dofs_per_cell);
      std::vector<double> strong_residual_tracer_x_variation(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> convective_velocity_x_variation(
        sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_moving_mesh(
        sd.dofs_per_cell);
      std::vector<Tensor<2, dim>> to_mult_by_grad_phi_u_i_moving_mesh(
        sd.dofs_per_cell);
      std::vector<double> p_x_tr_G_j(sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_p_i_moving_mesh(sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_phi_i_moving_mesh(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_grad_phi_phi_i_moving_mesh(
        sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_mu_i_moving_mesh(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_grad_phi_mu_i_moving_mesh(
        sd.dofs_per_cell);

      const Tensor<2, dim> *grad_source_term_velocity;
      const Tensor<1, dim> *grad_source_pressure;
      const Tensor<1, dim> *grad_source_tracer;
      const Tensor<1, dim> *grad_source_potential;

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double mobility = sd.get_mobility(q);
        const double diffusive_flux_factor =
          sd.diffusive_flux_factor * mobility;
        const double JxW_moving = sd.JxW_moving[q];
        const double rho        = sd.density[q];
        const double eta        = sd.dynamic_viscosity[q];
        const double drhodphi   = sd.derivative_density_wrt_tracer[q];
        const double detadphi   = sd.derivative_dynamic_viscosity_wrt_tracer[q];

        const auto  &dudt       = sd.present_velocity_time_derivatives[q];
        const auto  &u          = sd.present_velocity_values[q];
        const auto  &grad_u     = sd.present_velocity_gradients[q];
        const auto  &sym_grad_u = sd.present_velocity_sym_gradients[q];
        const double div_u      = sd.present_velocity_divergence[q];
        const auto  &lap_u      = sd.present_velocity_laplacians[q];
        const auto  &grad_div_u = sd.present_velocity_grad_div[q];
        const auto  &source_u   = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }
        const auto u_dot_grad_u_ale = grad_u * u_conv;

        const auto  &p        = sd.present_pressure_values[q];
        const auto  &grad_p   = sd.present_pressure_gradients[q];
        const double source_p = sd.source_term_pressure[q];

        const auto phase_flux = sd.get_phase_flux(q);
        const auto  &dphidt         = sd.tracer_time_derivatives[q];
        const auto  &phi            = sd.tracer_values[q];
        const auto  &grad_phi       = sd.tracer_gradients[q];
        const auto  &mu             = sd.potential_values[q];
        const auto  &grad_mu        = sd.potential_gradients[q];
        const double source_phi     = sd.source_term_tracer[q];
        const double source_mu      = sd.source_term_potential[q];

        auto supg_velocity = u_conv;
        if constexpr (BaseType::with_stabilization)
          supg_velocity +=
            (sd.diffusive_flux_factor / rho) * phase_flux;

        const auto &phi_u            = sd.phi_u[q];
        const auto &grad_phi_u       = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u   = sd.sym_grad_phi_u[q];
        const auto &div_phi_u        = sd.div_phi_u[q];
        const auto &laplacian_phi_u  = sd.laplacian_phi_u[q];
        const auto &grad_div_phi_u   = sd.grad_div_phi_u[q];
        const auto &phi_p            = sd.phi_p[q];
        const auto &grad_phi_p       = sd.grad_phi_p[q];
        const auto &phi_phi          = sd.shape_phi[q];
        const auto &grad_phi_phi     = sd.grad_shape_phi[q];
        const auto &phi_mu           = sd.shape_mu[q];
        const auto &grad_phi_mu      = sd.grad_shape_mu[q];
        const auto &laplacian_phi_mu = sd.laplacian_shape_mu[q];

        //
        // Moving mesh related data
        //
        if constexpr (BaseType::with_moving_mesh)
        {
          phi_x             = &sd.phi_x[q];
          grad_phi_x_moving = &sd.grad_phi_x_moving[q];
          if constexpr (BaseType::with_ale_stabilization)
            hessian_phi_x_moving = &sd.hessian_phi_x_moving[q];

          grad_source_term_velocity = &sd.grad_source_velocity[q];
          grad_source_pressure      = &sd.grad_source_pressure[q];
          grad_source_tracer        = &sd.grad_source_tracer[q];
          grad_source_potential     = &sd.grad_source_potential[q];
        }


        // Mesh velocity divergence and its gradient (moving meshes only)
        double         div_w = 0.;
        Tensor<2, dim> grad_w;
        if constexpr (BaseType::with_moving_mesh)
          if (skew || conservative)
          {
            grad_w = sd.present_mesh_velocity_gradients[q];
            div_w  = trace(grad_w);
          }

        // Precompute shape functions-independent terms
        const auto to_mult_by_phi_u_i_phi_phi_j =
          (drhodphi * (dudt + u_dot_grad_u_ale - body_force) + grad_mu);

        const auto momentum_partial_residual =
          rho * (dudt - body_force + u_dot_grad_u_ale) + phi * grad_mu +
          source_u;
        const auto phi_partial_residual =
          conservative ? dphidt + phi * div_w + source_phi :
                         dphidt + u_conv * grad_phi + source_phi;

        // Skew-symmetric momentum: mass flux, energy-weighted BDF inertia and
        // the complete vector tested with v.
        Tensor<1, dim> mass_flux, skew_partial_residual;
        if (skew)
        {
          mass_flux = rho * u_conv + sd.diffusive_flux_factor * phase_flux;
          skew_partial_residual =
            sd.energy_time[q].inertia + 0.5 * (grad_u * mass_flux) +
            0.5 * rho * div_w * u - rho * body_force + phi * grad_mu + source_u;
        }
        const auto mu_partial_residual =
          sd.skew_symmetric_momentum ?
            mu - sigma_tilde_over_eps * sd.energy_time[q].potential_derivative +
              source_mu :
            mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) + source_mu;
        double inv_rho      = 0.;
        double dinvrho_dphi = 0.;

        if constexpr (BaseType::with_stabilization)
        {
          tau          = sd.tau_supg_velocity[q];
          inv_rho      = 1. / rho;
          dinvrho_dphi = -drhodphi * inv_rho * inv_rho;

          // Compute strong residual of the momentum equation, in force units
          // (i.e. multiplied by the density); see the rhs assembler for the
          // corresponding test operators.
          strong_residual_momentum = CHNSEnergy::momentum_residual<dim>(sd, q);


          strong_residual_momentum_variation_phi_phi =
            to_mult_by_phi_u_i_phi_phi_j - detadphi * (lap_u + grad_div_u);
        }

        if constexpr (BaseType::with_tracer_stabilization)
        {
          tau_tracer          = sd.tau_supg_tracer[q];

          // Compute strong residual of the tracer equation
          strong_residual_tracer =
            dphidt + u_conv * grad_phi - sd.get_phase_diffusion(q) +
            source_phi + (sd.conservative_tracer_transport ? phi * div_u : 0.);
        }

        // Clipping phi makes the density constant outside [-1, 1].
        const double supg_dinvrho_dphi =
          !sd.positive_density_extension && sd.tracer_limiter(phi) != phi ?
            0. :
            dinvrho_dphi;

        CahnHilliard::ProfileCorrectionLinearization<dim> correction;
        if (!sd.profile_correction_flux.empty())
          CahnHilliard::evaluate_profile_correction(
            sd.reconstructed_profile[q].value,
            sd.reconstructed_profile[q].gradient(),
            sd.needs_phase_diffusion() ? sd.reconstructed_profile[q].hessian() :
                                         Tensor<2, dim>(),
            sd.epsilon,
            sd.needs_phase_diffusion(),
            &correction);

        // Precompute quantities depending only on j
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
        {
          const auto &phi_u_j       = phi_u[j];
          const auto &grad_phi_u_j  = grad_phi_u[j];
          const auto &grad_phi_mu_j = grad_phi_mu[j];

          double dM = 0.;
          if (sd.variable_mobility)
          {
            const auto &evaluation = sd.mobility_evaluation[q];
            dM = evaluation.derivative_wrt_tracer * phi_phi[j] +
                 evaluation.derivative_wrt_velocity * phi_u_j +
                 evaluation.derivative_wrt_tracer_gradient * grad_phi_phi[j];
            mobility_variation[j] = dM;
          }
          CahnHilliard::ProfileCorrection<dim> dcorrection;
          if (!sd.profile_correction_flux.empty())
          {
            const double inverse_derivative =
              sd.profile_nodal_inverse[j].derivative;
            const auto dprofile = sd.reconstructed_profile[q].variation(
              inverse_derivative * phi_phi[j],
              inverse_derivative * grad_phi_phi[j],
              sd.needs_phase_diffusion() ?
                inverse_derivative * sd.hessian_shape_phi[q][j] :
                Tensor<2, dim>(),
              sd.needs_phase_diffusion());
            dcorrection = correction.variation(dprofile.value,
                                               dprofile.gradient,
                                               dprofile.hessian);
            dcorrection.flux *= sd.profile_correction_coefficient;
            dcorrection.divergence *= sd.profile_correction_coefficient;
            correction_flux_variation[j] = dcorrection.flux;
          }
          if (additional_flux)
            additional_momentum_variation[j] =
              sd.diffusive_flux_factor *
              (grad_u * (dM * grad_mu + dcorrection.flux));

          to_mult_by_phi_u_i_momentum[j] =
            rho *
              (bdf_c0 * phi_u_j + grad_phi_u_j * u_conv + grad_u * phi_u_j) +
            sd.diffusive_flux_factor * grad_phi_u_j * phase_flux;

          to_mult_by_phi_u_i_potential[j] =
            diffusive_flux_factor * grad_u * grad_phi_mu_j +
            phi * grad_phi_mu_j;

          phi_u_j_x_grad_phi[j] = phi_u_j * grad_phi;

          to_mult_by_phi_phi_i[j] =
            bdf_c0 * phi_phi[j] + u_conv * grad_phi_phi[j];

          if (skew)
          {
            // Variation of the mass flux m = rho u_conv + J w.r.t. u, phi, mu
            const auto dm =
              rho * phi_u_j + drhodphi * phi_phi[j] * u_conv +
              sd.diffusive_flux_factor *
                (mobility * grad_phi_mu_j + dM * grad_mu + dcorrection.flux);
            mass_flux_variation[j] = dm;

            // Differentiate the selected BDF1/BDF2 inertial term.
            skew_momentum_variation[j] =
              sd.energy_time[q].inertia_velocity_derivative * phi_u_j +
              sd.energy_time[q].inertia_phase_derivative * phi_phi[j] +
              0.5 * (grad_phi_u_j * mass_flux + grad_u * dm) +
              0.5 * div_w * (rho * phi_u_j + drhodphi * phi_phi[j] * u) -
              drhodphi * phi_phi[j] * body_force + phi_phi[j] * grad_mu +
              phi * grad_phi_mu_j;
          }

          if (conservative)
          {
            // -(phi u_conv, grad psi) + (dphi/dt + phi div(w), psi)
            conservative_tracer_variation[j] =
              bdf_c0 * phi_phi[j] + phi_phi[j] * div_w;
            conservative_tracer_grad_variation[j] =
              -(phi * phi_u_j + phi_phi[j] * u_conv);
          }

          if constexpr (BaseType::with_stabilization)
          {
            // Tau stays frozen, but the direction depends on u, phi and mu.
            supg_velocity_variation[j] =
              phi_u_j +
              sd.diffusive_flux_factor * inv_rho *
                (dM * grad_mu + dcorrection.flux) +
              diffusive_flux_factor * inv_rho * grad_phi_mu_j +
              sd.diffusive_flux_factor * supg_dinvrho_dphi * phi_phi[j] *
                phase_flux;
          }

          if constexpr (BaseType::with_stabilization)
          {
            // As in the stabilized NS assembler, tau is kept constant in the
            // Newton Jacobian; only the residual and test operator are
            // linearized.

            // Variation w.r.t. velocity and pressure
            strong_residual_momentum_variation[j] =
              to_mult_by_phi_u_i_momentum[j] + grad_phi_p[j] -
              eta * (laplacian_phi_u[j] + grad_div_phi_u[j]) -
              2. * detadphi * (sym_grad_phi_u[j] * grad_phi);

            // Variation w.r.t. tracer
            strong_residual_momentum_variation[j] +=
              phi_phi[j] * strong_residual_momentum_variation_phi_phi -
              2. * detadphi * (sym_grad_u * grad_phi_phi[j]);

            // Variation w.r.t. potential
            strong_residual_momentum_variation[j] +=
              diffusive_flux_factor * grad_u * grad_phi_mu_j +
              phi * grad_phi_mu_j;
            if (additional_flux)
              strong_residual_momentum_variation[j] +=
                additional_momentum_variation[j];
          }

          if (sd.needs_phase_diffusion())
          {
            strong_residual_tracer_variation[j] =
              bdf_c0 * phi_phi[j] + phi_u_j * grad_phi +
              u_conv * grad_phi_phi[j] - mobility * laplacian_phi_mu[j] -
              dcorrection.divergence;
            if (sd.variable_mobility)
            {
              const auto dgrad_M =
                sd.mobility_gradient_variation(q,
                                               phi_phi[j],
                                               phi_u_j,
                                               grad_phi_u_j,
                                               grad_phi_phi[j],
                                               sd.gradient_dependent_mobility ?
                                                 sd.hessian_shape_phi[q][j] :
                                                 Tensor<2, dim>());
              strong_residual_tracer_variation[j] -=
                dM * sd.potential_laplacians[q] + dgrad_M * grad_mu +
                sd.mobility_gradient[q] * grad_phi_mu_j;
            }
          }

          if (skew && sd.needs_phase_diffusion())
          {
            const double d_diffusion =
              bdf_c0 * phi_phi[j] + phi_u_j * grad_phi +
              u_conv * grad_phi_phi[j] - strong_residual_tracer_variation[j];
            const double transport_divergence =
              drhodphi * (grad_phi * u_conv) + rho * div_u +
              sd.diffusive_flux_factor * sd.get_phase_diffusion(q);
            const double d_transport_divergence =
              drhodphi * (grad_phi_phi[j] * u_conv + grad_phi * phi_u_j +
                          phi_phi[j] * div_u) +
              rho * div_phi_u[j] + sd.diffusive_flux_factor * d_diffusion;
            if constexpr (BaseType::with_stabilization)
              strong_residual_momentum_variation[j] +=
                (sd.energy_time[q].inertia_velocity_derivative - rho * bdf_c0) *
                  phi_u_j +
                (sd.energy_time[q].inertia_phase_derivative - drhodphi * dudt) *
                  phi_phi[j] +
                0.5 *
                  (d_transport_divergence * u + transport_divergence * phi_u_j);
            strong_residual_tracer_variation[j] +=
              phi_phi[j] * div_u + phi * div_phi_u[j];
          }

          // Variations w.r.t. mesh position
          if constexpr (BaseType::with_moving_mesh)
          {
            const auto  &phi_x_j     = (*phi_x)[j];
            const auto  &G           = (*grad_phi_x_moving)[j];
            const auto   transpose_G = transpose(G);
            const double trG         = trace(G);

            const auto grad_u_x_G_j = grad_u * G;
            const auto   dgrad_phi_x  = -(transpose_G * grad_phi);
            const double dM_x =
              sd.variable_mobility ?
                sd.mobility_evaluation[q].derivative_wrt_tracer_gradient *
                  dgrad_phi_x :
                0.;

            Tensor<2, dim> dhess_phi_x;
            if (sd.needs_phase_diffusion())
              if (sd.gradient_dependent_mobility)
              {
                const auto &H = sd.tracer_hessians[q];
                const auto &K = (*hessian_phi_x_moving)[j];
                dhess_phi_x   = -(transpose_G * H + H * G);
                for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                    for (unsigned int c = 0; c < dim; ++c)
                      dhess_phi_x[a][b] -= grad_phi[c] * K[c][a][b];
              }
            CahnHilliard::ProfileCorrection<dim> dcorrection_x;
            if (!sd.profile_correction_flux.empty())
            {
              const auto    &profile      = sd.reconstructed_profile[q];
              const auto     grad_profile = profile.gradient();
              Tensor<2, dim> dhess_profile;
              if (sd.needs_phase_diffusion())
              {
                const auto  H = profile.hessian();
                const auto &K = (*hessian_phi_x_moving)[j];
                dhess_profile = -(transpose_G * H + H * G);
                for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                    for (unsigned int c = 0; c < dim; ++c)
                      dhess_profile[a][b] -= grad_profile[c] * K[c][a][b];
              }
              dcorrection_x = correction.variation(0.,
                                                   -transpose_G * grad_profile,
                                                   dhess_profile);
              dcorrection_x.flux *= sd.profile_correction_coefficient;
              dcorrection_x.divergence *= sd.profile_correction_coefficient;
            }

            p_x_tr_G_j[j] = p * trG;

            /**
             * Weak laplacian-like products (e.g., grad_phi_mu \cdot grad_phi in
             * the potential equation) vary like this with the mesh position:
             *
             * delta_x_j ((grad_phi_mu \cdot grad_phi) * dx) =
             *   (-G_j^T * grad_phi_mu) * grad_phi + grad_phi_mu * (-G_j^T *
             *   grad_phi)
             *     + (grad_phi_mu * grad_phi) * trace(G_j),
             *
             * which can be written as
             *   grad_phi_mu * ((-G_j - G_j^T + trace(G_j) * I) * grad_phi
             * = grad_phi_mu * ((- 2*sym(G_j) + trace(G_j) * I) * grad_phi.
             *
             * The quantity below is the one in parentheses.
             */
            const Tensor<2, dim> val =
              trG * identity_tensor - 2. * symmetrize(G);

            // The correction is evaluated on the current physical mesh.
            // Differentiate its flux, the test gradient and the volume measure.
            // Variation of momentum
            to_mult_by_phi_u_i_moving_mesh[j] =
              momentum_partial_residual * trG +
              rho * (grad_u * (-bdf_c0 * phi_x_j - G * u_conv)) -
              phi * transpose_G * grad_mu +
              diffusive_flux_factor * grad_u * val * grad_mu +
              sd.diffusive_flux_factor * dM_x * (grad_u * grad_mu);
            Tensor<1, dim> weak_correction_x;
            if (!sd.profile_correction_flux.empty())
            {
              weak_correction_x = dcorrection_x.flux +
                                  trG * sd.profile_correction_flux[q] -
                                  G * sd.profile_correction_flux[q];
              to_mult_by_phi_u_i_moving_mesh[j] +=
                sd.diffusive_flux_factor * (grad_u * weak_correction_x);
            }

            to_mult_by_grad_phi_u_i_moving_mesh[j] =
              p * transpose_G +
              2. * eta *
                (sym_grad_u * (trG * identity_tensor - transpose_G) -
                 symmetrize(grad_u_x_G_j));

            // Variation of continuity
            to_mult_by_phi_p_i_moving_mesh[j] =
              trace(grad_u_x_G_j) + (-div_u + source_p) * trG;

            // Variation of tracer
            to_mult_by_phi_phi_i_moving_mesh[j] =
              phi_partial_residual * trG - bdf_c0 * (phi_x_j * grad_phi) -
              u_conv * (transpose_G * grad_phi);

            to_mult_by_grad_phi_phi_i_moving_mesh[j] =
              mobility * (val * grad_mu) + dM_x * grad_mu + weak_correction_x;

            // Variation of the mesh velocity divergence:
            // grad(w) -> grad(w) + bdf_c0 G - grad(w) G.
            const double ddiv_w_dx = bdf_c0 * trG - trace(grad_w * G);

            if (conservative)
            {
              // (dphi/dt + phi div(w), psi) and -(phi u_conv, grad psi), with
              // u_conv -> u_conv - bdf_c0 phi_x and grad psi -> -G^T grad psi.
              to_mult_by_phi_phi_i_moving_mesh[j] =
                phi_partial_residual * trG + phi * ddiv_w_dx;
              to_mult_by_grad_phi_phi_i_moving_mesh[j] +=
                phi * (bdf_c0 * phi_x_j + G * u_conv - trG * u_conv);
            }

            if (skew)
            {
              const auto dmass_flux_dx =
                rho * (-bdf_c0 * phi_x_j) +
                sd.diffusive_flux_factor *
                  (mobility * (-(transpose_G * grad_mu)) + dM_x * grad_mu +
                   dcorrection_x.flux);
              skew_momentum_x_variation[j] =
                skew_partial_residual * trG +
                0.5 * (-(grad_u_x_G_j * mass_flux) + grad_u * dmass_flux_dx) +
                0.5 * rho * ddiv_w_dx * u - phi * transpose_G * grad_mu;
              skew_momentum_x_variation[j] +=
                (*grad_source_term_velocity) * phi_x_j;
              skew_mass_flux_x_variation[j] =
                -(G * mass_flux) + dmass_flux_dx + trG * mass_flux;
            }

            // Variation of potential
            to_mult_by_phi_mu_i_moving_mesh[j] = mu_partial_residual * trG;

            to_mult_by_grad_phi_mu_i_moving_mesh[j] =
              -sigma_tilde_times_eps * (val * grad_phi);

            to_mult_by_phi_u_i_moving_mesh[j] +=
              (*grad_source_term_velocity) * phi_x_j;
            to_mult_by_phi_p_i_moving_mesh[j] +=
              (*grad_source_pressure) * phi_x_j;
            to_mult_by_phi_phi_i_moving_mesh[j] +=
              (*grad_source_tracer) * phi_x_j;
            to_mult_by_phi_mu_i_moving_mesh[j] +=
              (*grad_source_potential) * phi_x_j;

            // x-variation of the SUPG/PSPG strong residuals (frozen tau).
            // Gradients transform as grad -> grad - G^T grad, the velocity
            // gradient as grad_u -> grad_u - grad_u G, and the ALE convective
            // velocity as u_conv -> u_conv - bdf_c0 phi_x.
            if constexpr (BaseType::with_ale_stabilization)
            {
              const auto du_conv_dx   = -bdf_c0 * phi_x_j;
              const auto dgrad_u_dx   = -grad_u_x_G_j; // -(grad_u * G)
              const auto dgrad_mu_dx  = -(transpose_G * grad_mu);
              const auto dgrad_phi_dx = -(transpose_G * grad_phi);

              convective_velocity_x_variation[j] = du_conv_dx;
              if constexpr (BaseType::with_stabilization)
                supg_velocity_x_variation[j] =
                  du_conv_dx + diffusive_flux_factor * inv_rho * dgrad_mu_dx +
                  sd.diffusive_flux_factor * inv_rho *
                    (dM_x * grad_mu + dcorrection_x.flux);

              // Variation of the momentum stabilization term w.r.t. position
              if constexpr (BaseType::with_stabilization)
              {
                const auto &H          = sd.present_velocity_hessians[q];
                const auto &K          = (*hessian_phi_x_moving)[j];
                const auto  dgrad_p_dx = -(transpose_G * grad_p);

                // d(lap_u + grad_div_u)/dx_j, with H[c][i][k] = d^2 u_c.
                // Here dx_j is the scalar mesh DoF coefficient, not a spatial
                // coordinate: this directional variation is still a vector.
                Tensor<1, dim> dlap_plus_graddiv_u_dx;
                for (unsigned int c = 0; c < dim; ++c)
                  for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int k = 0; k < dim; ++k)
                    {
                      double dH = 0.;
                      for (unsigned int m = 0; m < dim; ++m)
                        dH -= G[m][i] * H[c][m][k] + G[m][k] * H[c][i][m] +
                              K[m][i][k] * grad_u[c][m];
                      if (i == k)
                        dlap_plus_graddiv_u_dx[c] += dH;
                      if (i == c)
                        dlap_plus_graddiv_u_dx[k] += dH;
                    }

                const auto dsym_grad_u_dx = symmetrize(dgrad_u_dx);
                const auto dviscous_divergence_dx =
                  eta * dlap_plus_graddiv_u_dx +
                  2. * detadphi *
                    (dsym_grad_u_dx * grad_phi + sym_grad_u * dgrad_phi_dx);

                auto ddiffusive_flux_dx =
                  diffusive_flux_factor *
                    (dgrad_u_dx * grad_mu + grad_u * dgrad_mu_dx) +
                  sd.diffusive_flux_factor * dM_x * (grad_u * grad_mu);
                if (!sd.profile_correction_flux.empty())
                  ddiffusive_flux_dx +=
                    sd.diffusive_flux_factor *
                    (dgrad_u_dx * sd.profile_correction_flux[q] +
                     grad_u * dcorrection_x.flux);
                const auto dcapillary_dx = phi * dgrad_mu_dx;

                strong_residual_momentum_x_variation[j] =
                  rho * (dgrad_u_dx * u_conv + grad_u * du_conv_dx) +
                  dgrad_p_dx + ddiffusive_flux_dx + dcapillary_dx -
                  dviscous_divergence_dx;
                strong_residual_momentum_x_variation[j] +=
                  (*grad_source_term_velocity) * phi_x_j;
              }

              // Variation of the tracer stabilization term w.r.t. position
              if (sd.needs_phase_diffusion())
              {
                const auto &h = sd.potential_hessians[q];
                const auto &K = (*hessian_phi_x_moving)[j];

                // d(laplacian mu)/dx_j = trace of the scalar hessian variation.
                double dlap_mu_dx = 0.;
                for (unsigned int i = 0; i < dim; ++i)
                  for (unsigned int k = 0; k < dim; ++k)
                    dlap_mu_dx -= G[k][i] * h[k][i] + G[k][i] * h[i][k] +
                                  K[k][i][i] * grad_mu[k];

                strong_residual_tracer_x_variation[j] = du_conv_dx * grad_phi +
                                                        u_conv * dgrad_phi_dx -
                                                        mobility * dlap_mu_dx;
                strong_residual_tracer_x_variation[j] -=
                  dcorrection_x.divergence;
                if (sd.variable_mobility)
                {
                  const auto dgrad_M =
                    sd.mobility_gradient_variation(q,
                                                   0.,
                                                   Tensor<1, dim>(),
                                                   dgrad_u_dx,
                                                   dgrad_phi_dx,
                                                   dhess_phi_x);
                  strong_residual_tracer_x_variation[j] -=
                    dM_x * sd.potential_laplacians[q] + dgrad_M * grad_mu +
                    sd.mobility_gradient[q] * dgrad_mu_dx;
                }
                strong_residual_tracer_x_variation[j] +=
                  (*grad_source_tracer) * phi_x_j;
                if (skew)
                {
                  double d_diffusion = du_conv_dx * grad_phi +
                                       u_conv * dgrad_phi_dx -
                                       strong_residual_tracer_x_variation[j];
                  d_diffusion += (*grad_source_tracer) * phi_x_j;
                  const double ddiv_u = -trace(grad_u * G);
                  const double d_transport_divergence =
                    drhodphi * (dgrad_phi_dx * u_conv + grad_phi * du_conv_dx) +
                    rho * ddiv_u + sd.diffusive_flux_factor * d_diffusion;
                  if constexpr (BaseType::with_stabilization)
                    strong_residual_momentum_x_variation[j] +=
                      0.5 * d_transport_divergence * u;
                  strong_residual_tracer_x_variation[j] += phi * ddiv_u;
                }
              }
            }
          }
        }

        /**
         * Assemble the local matrix.
         * The loops over the j degrees of freedom are repeated for each
         * assembled equation: this removes the tests over i inside the j
         * loop, and is (ever so slightly) more efficient. Looping only over
         * the coupled dofs for each variable (obtained by creating a set
         * first in the scratch, for instance) does not yield any tremendous
         * additional gain though.
         */

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i = sd.components[i];
          const bool         i_is_x = this->ordering.is_position(comp_i);
          if (i_is_x)
            continue;

          // Iterator to the current matrix row
          auto matrix_row = local_matrix[i];

          const auto &phi_u_i          = phi_u[i];
          const auto &grad_phi_u_i     = grad_phi_u[i];
          const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
          const auto &div_phi_u_i      = div_phi_u[i];
          const auto &phi_p_i          = phi_p[i];
          const auto &grad_phi_p_i     = grad_phi_p[i];
          const auto &phi_phi_i        = phi_phi[i];
          const auto &grad_phi_phi_i   = grad_phi_phi[i];
          const auto &phi_mu_i         = phi_mu[i];
          const auto &grad_phi_mu_i    = grad_phi_mu[i];

          /**
           * Momentum equation
           */
          if (this->ordering.is_velocity(comp_i))
          {
            if constexpr (BaseType::with_stabilization)
            {
              supg_velocity_dot_grad_phi_u_i = grad_phi_u_i * supg_velocity;
              residual_dot_grad_phi_u_i =
                strong_residual_momentum * grad_phi_u_i;
            }

            // Skew-symmetric term -1/2 u . (grad(v_i) m): grad(v_i) m and
            // grad(v_i)^T u.
            Tensor<1, dim> grad_phi_u_i_mass_flux, grad_phi_u_i_T_u;
            if (skew)
            {
              grad_phi_u_i_mass_flux = grad_phi_u_i * mass_flux;
              grad_phi_u_i_T_u       = transpose(grad_phi_u_i) * u;
            }

            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j   = sd.components[j];
              const bool         j_is_u   = this->ordering.is_velocity(comp_j);
              const bool         j_is_p   = this->ordering.is_pressure(comp_j);
              const bool         j_is_phi = this->ordering.is_tracer(comp_j);
              const bool         j_is_mu  = this->ordering.is_potential(comp_j);
              const bool         j_is_x   = this->ordering.is_position(comp_j);

              const auto &phi_phi_j = phi_phi[j];

              // Account for the pressure gradient when initializing
              double local_matrix_ij = j_is_p ? -div_phi_u_i * phi_p[j] : 0.;

              if (j_is_u)
              {
                if (!skew)
                  local_matrix_ij += phi_u_i * to_mult_by_phi_u_i_momentum[j];

                // Diffusion: 2. * eta * scalar_product(sym_grad_phi_u[j],
                // sym_grad_phi_u_i), explicited for the symmetric gradient of
                // Lagrange shape functions.
                const auto &gui = grad_phi_u_i[comp_i - u_lower];
                const auto &guj = grad_phi_u[j][comp_j - u_lower];
                local_matrix_ij +=
                  eta * (gui[comp_j - u_lower] * guj[comp_i - u_lower]);
                if (comp_i == comp_j)
                  local_matrix_ij += eta * gui * guj;
              }
              else if (j_is_phi)
                local_matrix_ij +=
                  phi_phi_j *
                  ((skew ? 0. : phi_u_i * to_mult_by_phi_u_i_phi_phi_j) +
                   2. * detadphi *
                     scalar_product(sym_grad_phi_u_i, sym_grad_u));
              else if (j_is_mu && !skew)
                local_matrix_ij += phi_u_i * to_mult_by_phi_u_i_potential[j];
              if (additional_flux && !j_is_x && !skew)
                local_matrix_ij += phi_u_i * additional_momentum_variation[j];
              if (skew && !j_is_x)
                local_matrix_ij +=
                  phi_u_i * skew_momentum_variation[j] -
                  0.5 * (phi_u[j] * grad_phi_u_i_mass_flux +
                         grad_phi_u_i_T_u * mass_flux_variation[j]);

              if constexpr (BaseType::with_stabilization)
              {
                // SUPG stabilization : variation w.r.t. u, p, phi and mu
                local_matrix_ij +=
                  tau * (strong_residual_momentum_variation[j] *
                           supg_velocity_dot_grad_phi_u_i +
                         residual_dot_grad_phi_u_i *
                           supg_velocity_variation[j]);
              }

              if constexpr (BaseType::with_moving_mesh)
              {
                if (j_is_x)
                {
                  // Momentum : variation w.r.t. moving mesh position

                  // Simplification of the double contraction grad_phi_u_i : T,
                  // with T = to_mult_by_grad_phi_u_i_moving_mesh[j], valid
                  // for vector-valued Lagrange shape functions.
                  const auto &grad_phi_u_i_row = grad_phi_u_i[comp_i - u_lower];
                  const auto &t_row =
                    to_mult_by_grad_phi_u_i_moving_mesh[j][comp_i - u_lower];

                  local_matrix_ij +=
                    -div_phi_u_i * p_x_tr_G_j[j] + grad_phi_u_i_row * t_row;
                  if (skew)
                    local_matrix_ij +=
                      phi_u_i * skew_momentum_x_variation[j] -
                      0.5 * grad_phi_u_i_T_u * skew_mass_flux_x_variation[j];
                  else
                    local_matrix_ij +=
                      phi_u_i * to_mult_by_phi_u_i_moving_mesh[j];

                  // SUPG : variation w.r.t. mesh position (frozen tau). The
                  // trace term accounts for the variation of JxW_moving, which
                  // multiplies local_matrix_ij below.
                  if constexpr (BaseType::with_stabilization)
                  {
                    const auto &G = sd.grad_phi_x_moving[q][j];
                    local_matrix_ij +=
                      tau * (supg_velocity_dot_grad_phi_u_i *
                               strong_residual_momentum_x_variation[j] +
                             strong_residual_momentum *
                               (-(grad_phi_u_i * G) * supg_velocity +
                                grad_phi_u_i * supg_velocity_x_variation[j] +
                                supg_velocity_dot_grad_phi_u_i * trace(G)));
                  }
                }
              }

              // Increment local matrix
              matrix_row[j] += local_matrix_ij * JxW_moving;
            }
          }

          /**
           * Continuity equation
           */
          if (this->ordering.is_pressure(comp_i))
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j = sd.components[j];
              const bool         j_is_u = this->ordering.is_velocity(comp_j);
              const bool         j_is_x = this->ordering.is_position(comp_j);

              if (j_is_u)
                matrix_row[j] += -phi_p_i * div_phi_u[j] * JxW_moving;

              if constexpr (BaseType::with_stabilization)
                // PSPG stabilization : variation w.r.t. u and p
                matrix_row[j] +=
                  -tau *
                  ((inv_rho * strong_residual_momentum_variation[j] +
                    dinvrho_dphi * phi_phi[j] * strong_residual_momentum) *
                   grad_phi_p_i) *
                  JxW_moving;

              if constexpr (BaseType::with_moving_mesh)
              {
                if (j_is_x)
                {
                  // Continuity : variation w.r.t. x
                  matrix_row[j] +=
                    phi_p_i * to_mult_by_phi_p_i_moving_mesh[j] * JxW_moving;

                  // PSPG : variation w.r.t. mesh position (frozen tau).
                  if constexpr (BaseType::with_stabilization)
                  {
                    const auto &G = sd.grad_phi_x_moving[q][j];
                    // d(grad_phi_p_i)/dx_j = -(G^T grad_phi_p_i)
                    // Density depends on the tracer coefficients, held fixed
                    // in position columns; its tracer variation is above.
                    matrix_row[j] +=
                      -(tau * inv_rho *
                        (strong_residual_momentum_x_variation[j] *
                           grad_phi_p_i +
                         strong_residual_momentum *
                           (-grad_phi_p_i * G + grad_phi_p_i * trace(G)))) *
                      JxW_moving;
                  }
                }
              }
            }

          /**
           * Tracer equation
           */
          else if (this->ordering.is_tracer(comp_i))
          {
            if constexpr (BaseType::with_tracer_stabilization)
            {
              u_conv_dot_grad_phi_phi_i = u_conv * grad_phi_phi_i;
              residual_tracer_dot_grad_phi_phi_i =
                strong_residual_tracer * grad_phi_phi_i;
            }

            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j   = sd.components[j];
              const bool         j_is_u   = this->ordering.is_velocity(comp_j);
              const bool         j_is_phi = this->ordering.is_tracer(comp_j);
              const bool         j_is_mu  = this->ordering.is_potential(comp_j);
              const bool         j_is_x   = this->ordering.is_position(comp_j);

              const auto &grad_phi_mu_j = grad_phi_mu[j];

              if (conservative && (j_is_u || j_is_phi))
              {
                matrix_row[j] +=
                  (phi_phi_i * conservative_tracer_variation[j] +
                   grad_phi_phi_i * conservative_tracer_grad_variation[j]) *
                  JxW_moving;
              }
              else if (j_is_u)
              {
                matrix_row[j] += phi_phi_i * phi_u_j_x_grad_phi[j] * JxW_moving;
              }
              else if (j_is_phi)
              {
                matrix_row[j] +=
                  phi_phi_i * to_mult_by_phi_phi_i[j] * JxW_moving;
              }
              else if (j_is_mu)
              {
                matrix_row[j] +=
                  mobility * (grad_phi_mu_j * grad_phi_phi_i) * JxW_moving;
              }

              if (sd.variable_mobility && !j_is_x)
                matrix_row[j] += mobility_variation[j] *
                                 (grad_mu * grad_phi_phi_i) * JxW_moving;
              if (!sd.profile_correction_flux.empty() && !j_is_x)
                matrix_row[j] +=
                  (correction_flux_variation[j] * grad_phi_phi_i) * JxW_moving;

              if constexpr (BaseType::with_tracer_stabilization)
              {
                // Tracer SUPG stabilization : variation w.r.t. u and phi
                matrix_row[j] +=
                  tau_tracer *
                  (strong_residual_tracer_variation[j] *
                     u_conv_dot_grad_phi_phi_i +
                   residual_tracer_dot_grad_phi_phi_i * phi_u[j]) *
                  JxW_moving;
              }
              if constexpr (BaseType::with_moving_mesh)
              {
                // Tracer : variation w.r.t. x
                if (j_is_x)
                {
                  matrix_row[j] +=
                    (phi_phi_i * to_mult_by_phi_phi_i_moving_mesh[j] +
                     grad_phi_phi_i *
                       to_mult_by_grad_phi_phi_i_moving_mesh[j]) *
                    JxW_moving;

                  // Tracer-SUPG : variation w.r.t. mesh position (frozen tau).
                  if constexpr (BaseType::with_tracer_stabilization)
                  {
                    const auto &G = sd.grad_phi_x_moving[q][j];
                    matrix_row[j] +=
                      tau_tracer *
                      (u_conv_dot_grad_phi_phi_i *
                         strong_residual_tracer_x_variation[j] +
                       strong_residual_tracer *
                         (convective_velocity_x_variation[j] * grad_phi_phi_i -
                          u_conv * (grad_phi_phi_i * G) +
                          u_conv_dot_grad_phi_phi_i * trace(G))) *
                      JxW_moving;
                  }
                }
              }
            }
          }

          /**
           * Potential equation
           */
          else if (this->ordering.is_potential(comp_i))
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j   = sd.components[j];
              const bool         j_is_phi = this->ordering.is_tracer(comp_j);
              const bool         j_is_mu  = this->ordering.is_potential(comp_j);
              const bool         j_is_x   = this->ordering.is_position(comp_j);

              const auto &phi_phi_j      = phi_phi[j];
              const auto &grad_phi_phi_j = grad_phi_phi[j];
              const auto &phi_mu_j       = phi_mu[j];

              if (j_is_mu)
              {
                // Mass
                matrix_row[j] += phi_mu_i * phi_mu_j * JxW_moving;
              }
              else if (j_is_phi)
              {
                matrix_row[j] +=
                  (-sigma_tilde_over_eps * phi_mu_i * phi_phi_j *
                     (skew ? sd.energy_time[q].potential_second_derivative :
                             3. * phi * phi - 1.) -
                   sigma_tilde_times_eps * (grad_phi_mu_i * grad_phi_phi_j)) *
                  JxW_moving;
              }

              if constexpr (BaseType::with_moving_mesh)
              {
                // Potential : variation w.r.t. x
                if (j_is_x)
                {
                  matrix_row[j] +=
                    (phi_mu_i * to_mult_by_phi_mu_i_moving_mesh[j] +
                     grad_phi_mu_i * to_mult_by_grad_phi_mu_i_moving_mesh[j]) *
                    JxW_moving;
                }
              }
            }
        }
      }

      if constexpr (BaseType::with_enlarged)
        assemble_psi_equation_matrix<dim, BaseType::with_moving_mesh>(
          this->ordering, this->coupling_table, sd, local_matrix);
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

// Explicit instantiations
#include "incompressible_chns_assemblers.inst"
