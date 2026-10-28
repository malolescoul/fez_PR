#include <assembly/elasticity_assemblers.h>
#include <assembly/incompressible_chns_assemblers.h>
#include <chns_energy_diagnostics.h>
#include <compare_matrix.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <elasticity_solver.h>
#include <errors.h>
#include <incompressible_chns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <mesh_and_dof_tools.h>
#include <scratch_data.h>
#include <utilities.h>

#include <limits>

template <int dim, bool with_moving_mesh, bool with_enlarged>
bool CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  set_solver_specific_initial_mesh_position()
{
  if constexpr (!with_moving_mesh)
    return false;
  else
  {
    if (!this->param.cahn_hilliard.use_presolver)
      return false;

    AssertThrow(
      !this->param.with_tree_based_adaptation() &&
        !this->param.with_metric_based_adaptation(),
      ExcMessage(
        "The CHNS-ALE presolver currently requires a fixed reference mesh."));

    if (this->time_handler.current_time_iteration == 0)
    {
      auto presolver_param = this->param;
      // Presolving is stationary: failure must prevent the CHNS handoff, even
      // when the main simulation allows adaptive time-step rejection.
      presolver_param.time_integration.scheme =
        Parameters::TimeIntegration::Scheme::stationary;
      ElasticitySolver<dim> presolver(presolver_param,
                                      this->triangulation,
                                      with_enlarged);
      presolver.run();
      std::map<unsigned int, unsigned int> components;
      for (unsigned int d = 0; d < dim; ++d)
        components[d] = this->ordering->x_lower + d;
      if constexpr (with_enlarged)
        components[dim] = this->ordering->psi_lower;
      extract_subsolution<dim>(presolver.get_dof_handler(),
                               *this->dof_handler,
                               presolver.get_present_solution(),
                               this->newton_update,
                               components);
    }
    else
    {
      // Repeating the initial condition for BDF startup keeps its mesh
      // position and the reconstructed marker.
      for (const auto i : this->locally_owned_dofs)
        if (this->ordering->is_position(
              this->dofs_to_component[this->locally_relevant_dofs
                                        .index_within_set(i)]) ||
            (with_enlarged &&
             this->ordering->is_psi(
               this->dofs_to_component[this->locally_relevant_dofs
                                         .index_within_set(i)])))
          this->newton_update[i] = (*this->present_solution)[i];
    }
    this->newton_update.compress(VectorOperation::insert);
    if (this->time_handler.current_time_iteration == 0)
    {
      // Repeat the CHNS summary after the presolver's continuation and timers.
      this->pcout << "\nCHNS-ALE" << (with_enlarged ? " enlarged" : "")
                  << std::endl;
      if (this->param.mesh.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "Mesh info:\n dimension: " << dim
                    << "\n no. of cells: "
                    << this->triangulation->n_global_active_cells()
                    << std::endl;
      this->pcout << "Number of degrees of freedom: "
                  << this->dof_handler->n_dofs() << std::endl;
    }
    return true;
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::CHNSSolver(
  const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  if (param.cahn_hilliard.enable_energy_consistent_formulation)
  {
    using Scheme = Parameters::TimeIntegration::Scheme;
    AssertThrow(param.time_integration.scheme == Scheme::BDF1 ||
                  param.time_integration.scheme == Scheme::BDF2,
                ExcMessage(
                  "The energy-consistent formulation requires BDF1 or BDF2."));
    AssertThrow(param.time_integration.scheme != Scheme::BDF2 ||
                  (param.time_integration.bdfstart ==
                     Parameters::TimeIntegration::BDFStart::BDF1 &&
                   param.time_integration.bdf_starting_step_ratio > 0. &&
                   param.time_integration.bdf_starting_step_ratio < 1.),
                ExcMessage("Energy-consistent BDF2 requires bdf start method = "
                           "BDF1 and 0 < bdf start step ratio < 1."));
    AssertThrow(param.cahn_hilliard.epsilon_interface > 0. &&
                  param.cahn_hilliard.surface_tension > 0.,
                ExcMessage("Energy-consistent CHNS requires positive interface "
                           "thickness and surface tension."));
    AssertThrow(param.physical_properties.fluids[0].density > 0. &&
                  param.physical_properties.fluids[1].density > 0.,
                ExcMessage("Kinetic density must be positive in both fluids."));
  }
  if (param.cahn_hilliard.enable_profile_correction)
  {
    AssertThrow(!this->time_handler.is_steady(),
                ExcMessage("Profile correction currently requires transient "
                           "CHNS, CHNS-ALE or enlarged CHNS-ALE."));
    AssertThrow(param.cahn_hilliard.epsilon_interface > 0. &&
                  param.cahn_hilliard.surface_tension > 0.,
                ExcMessage("Profile correction requires positive interface "
                           "thickness and surface tension."));
  }
  if constexpr (with_enlarged)
  {
    // Enlarged ALE: same layout as the moving-mesh CHNS, with the extra psi
    // tracer appended after the potential. The psi field reuses the tracer FE
    // degree.
    if (param.finite_elements.use_quads)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                      dim),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree),
        FE_Q<dim>(param.finite_elements.tracer_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                      dim),
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(
          FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree));

    this->ordering = std::make_unique<ComponentOrderingCHNS<dim, true, true>>();
  }
  else if constexpr (with_moving_mesh)
  {
    if (param.finite_elements.use_quads)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                      dim),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                      dim),
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(
          FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering = std::make_unique<ComponentOrderingCHNS<dim, true>>();
  }
  else
  {
    if (param.finite_elements.use_quads)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                      dim),
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering = std::make_unique<ComponentOrderingCHNS<dim, false>>();
  }

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  if constexpr (with_moving_mesh)
    this->position_extractor =
      FEValuesExtractors::Vector(this->ordering->x_lower);

  tracer_extractor    = FEValuesExtractors::Scalar(this->ordering->phi_lower);
  potential_extractor = FEValuesExtractors::Scalar(this->ordering->mu_lower);
  if constexpr (with_enlarged)
    psi_extractor = FEValuesExtractors::Scalar(this->ordering->psi_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  if constexpr (with_moving_mesh)
    this->position_mask = fe->component_mask(this->position_extractor);

  tracer_mask    = fe->component_mask(tracer_extractor);
  potential_mask = fe->component_mask(potential_extractor);
  if constexpr (with_enlarged)
    psi_mask = fe->component_mask(psi_extractor);

  this->field_names_and_masks["velocity"]  = this->velocity_mask;
  this->field_names_and_masks["pressure"]  = this->pressure_mask;
  this->field_names_and_masks["tracer"]    = this->tracer_mask;
  this->field_names_and_masks["potential"] = this->potential_mask;
  if constexpr (with_enlarged)
    this->field_names_and_masks["psi"] = this->psi_mask;

  /**
   * Create the initial condition functions
   */
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);
  this->param.initial_conditions.create_initial_chns_tracer(
    this->ordering->phi_lower, this->ordering->n_components);

  // Assign the exact solution
  this->exact_solution = std::make_shared<
    CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSolution>(
    this->time_handler.current_time, *this->ordering, param.mms);

  if (param.mms_param.enable)
  {
    // Create the MMS source term function and override source terms
    this->source_terms = std::make_shared<
      CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSourceTerm>(
      this->time_handler.current_time, *this->ordering, param);

    // Create entry in error handler for tracer and potential
    for (auto &[norm, handler] : this->error_handlers)
    {
      handler.create_entry("phi");
      handler.create_entry("mu");
      if constexpr (with_enlarged)
        handler.create_entry("psi");
    }
  }
  else
  {
    this->source_terms = std::make_shared<
      CHNSSolver<dim, with_moving_mesh, with_enlarged>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSourceTerm::
  vector_value(const Point<dim> &p, Vector<double> &values) const
{
  const double phi          = mms.exact_tracer->value(p);
  const bool   energy_form =
    cahn_hilliard_param.enable_energy_consistent_formulation;
  const double filtered_phi =
    energy_form ? CahnHilliard::tracer_limiter(phi) : phi;
  const double limiter_derivative = energy_form && std::abs(phi) > 1. ? 0. : 1.;
  const double rho0         = physical_properties.fluids[0].density;
  const double rho1         = physical_properties.fluids[1].density;
  const double rho = CahnHilliard::linear_mixing(phi, rho0, rho1);
  const double kinetic_rho =
    CahnHilliard::linear_mixing(filtered_phi, rho0, rho1);
  const double eta0 = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1 = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  const double eta  = CahnHilliard::linear_mixing(filtered_phi, eta0, eta1);
  // const double drhodphi =
  //   CahnHilliard::linear_mixing_derivative(filtered_phi, rho0, rho1);
  const double detadphi =
    limiter_derivative *
    CahnHilliard::linear_mixing_derivative(filtered_phi, eta0, eta1);
  const double epsilon = cahn_hilliard_param.epsilon_interface;
  const double sigma_tilde =
    3. / (2. * sqrt(2.)) * cahn_hilliard_param.surface_tension;
  const double sigma_tilde_over_eps  = sigma_tilde / epsilon;
  const double sigma_tilde_times_eps = sigma_tilde * epsilon;
  const auto  &body_force            = physical_properties.body_force;

  Tensor<1, dim> u, dudt_eulerian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
  }

  // Use convention (grad_u)_ij := dvj/dxi
  Tensor<2, dim> grad_u      = mms.exact_velocity->gradient_vj_xi(p);
  Tensor<1, dim> lap_u       = mms.exact_velocity->vector_laplacian(p);
  Tensor<1, dim> grad_div_u  = mms.exact_velocity->grad_div(p);
  Tensor<1, dim> grad_p      = mms.exact_pressure->gradient(p);
  Tensor<1, dim> uDotGradu   = u * grad_u;
  Tensor<1, dim> grad_mu     = mms.exact_potential->gradient(p);
  Tensor<1, dim> grad_phi    = mms.exact_tracer->gradient(p);
  const auto     evaluate_mobility =
    CahnHilliard::get_mobility_evaluation_function(cahn_hilliard_param);
  const auto mobility_data =
    evaluate_mobility(cahn_hilliard_param,
                      phi,
                      u,
                      grad_phi,
                      CahnHilliard::get_adaptive_mobility_scaling(
                        cahn_hilliard_param));
  const double M = mobility_data.value;
  // grad_u here is transposed relative to the assembly convention.
  Tensor<1, dim> grad_M =
    mobility_data.derivative_wrt_tracer * grad_phi +
    grad_u * mobility_data.derivative_wrt_velocity +
    mms.exact_tracer->hessian(p) * mobility_data.derivative_wrt_tracer_gradient;
  CahnHilliard::ProfileCorrection<dim> correction;
  if (cahn_hilliard_param.enable_profile_correction)
    correction = CahnHilliard::evaluate_profile_correction<dim>(
      phi, grad_phi, mms.exact_tracer->hessian(p), epsilon);
  Tensor<1, dim> J_flux =
    0.5 * (rho1 - rho0) *
    (M * grad_mu + profile_correction_coefficient * correction.flux);
  Tensor<1, dim> div_viscous = (eta * (lap_u + grad_div_u) +
                                2. * detadphi * grad_phi * symmetrize(grad_u));

  // Navier-Stokes momentum (velocity) source term
  Tensor<1, dim> f = -(rho * (dudt_eulerian + uDotGradu - body_force) +
                       J_flux * grad_u + grad_p - div_viscous + phi * grad_mu);
  if (cahn_hilliard_param.enable_energy_consistent_formulation)
  {
    const double beta  = 0.5 * (rho0 - rho1);
    const double div_Q = M * mms.exact_potential->laplacian(p) +
                         grad_M * grad_mu +
                         profile_correction_coefficient * correction.divergence;
    Tensor<1, dim> mesh_velocity;
    if constexpr (with_moving_mesh)
      if (std::abs(phi) > 1.)
      {
        // Mesh position is prescribed in reference coordinates, whereas the
        // fluid MMS is evaluated at the physical point p. Invert the exact
        // map to evaluate its velocity when the two density extensions differ.
        Point<dim> reference = p;
        bool       converged = false;
        for (unsigned int iteration = 0; iteration < 30; ++iteration)
        {
          Tensor<1, dim> residual;
          Tensor<2, dim> jacobian;
          for (unsigned int d = 0; d < dim; ++d)
          {
            residual[d] = mms.exact_mesh_position->value(reference, d) - p[d];
            jacobian[d] = mms.exact_mesh_position->gradient(reference, d);
          }
          if (residual.norm() <= 1.e-12 * (1. + p.norm()))
          {
            converged = true;
            break;
          }
          reference -= invert(jacobian) * residual;
        }
        AssertThrow(converged,
                    ExcMessage(
                      "Cannot invert the exact ALE map for the energy MMS."));
        for (unsigned int d = 0; d < dim; ++d)
          mesh_velocity[d] =
            mms.exact_mesh_position->time_derivative(reference, d);
      }
    const double phase_rate = mms.exact_tracer->time_derivative(p);
    const double mass_residual =
      beta * (limiter_derivative * phase_rate + u * grad_phi - div_Q +
              (limiter_derivative - 1.) * (mesh_velocity * grad_phi)) +
      rho * mms.exact_velocity->divergence(p);
    f -= (kinetic_rho - rho) * (dudt_eulerian + mesh_velocity * grad_u) +
         0.5 * mass_residual * u;
  }
  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms).
  values[p_lower] = mms.exact_velocity->divergence(p);

  if constexpr (with_moving_mesh)
  {
    // Pseudosolid (mesh position) source term
    Tensor<1, dim> f_PS =
      mms.exact_mesh_position->divergence_elastic_stress_tensor(
        physical_properties.pseudosolids[0], p);

    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = f_PS[d];
  }

  // Tracer source term
  const double dphidt = mms.exact_tracer->time_derivative(p);
  const double lap_mu = mms.exact_potential->laplacian(p);
  values[phi_lower] = -(dphidt + u * grad_phi - M * lap_mu - grad_M * grad_mu -
                        profile_correction_coefficient * correction.divergence);

  if (cahn_hilliard_param.enable_energy_consistent_formulation)
    values[phi_lower] -= phi * mms.exact_velocity->divergence(p);

  // Potential source term
  const double mu      = mms.exact_potential->value(p);
  const double lap_phi = mms.exact_tracer->laplacian(p);
  values[mu_lower]     = -(mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) +
                       sigma_tilde_times_eps * lap_phi);
  // Enlarged (psi) Helmholtz reconstruction source term. The strong form is
  // psi - phi - mu_correction - L^2 lap(psi) = -source, so the manufactured
  // source is the negative of that residual evaluated at the exact solution.
  if constexpr (with_enlarged)
  {
    const double psi     = mms.exact_psi->value(p);
    const double lap_psi = mms.exact_psi->laplacian(p);
    const double L = cahn_hilliard_param.psi_interface_width_factor * epsilon;
    const double length_scale_sq = L * L;
    const double correction_prefactor =
      Assembly::IncompressibleCHNS::compute_psi_mu_correction_prefactor(
        cahn_hilliard_param.psi_mu_correction_factor,
        sigma_tilde,
        epsilon,
        length_scale_sq);
    const double psi_mu_correction =
      correction_prefactor *
      Assembly::IncompressibleCHNS::psi_mu_correction_eta(phi) * mu;
    values[psi_lower] =
      -(psi - phi - psi_mu_correction - length_scale_sq * lap_psi);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  update_simulation_parameters(const unsigned int fixed_point_iteration)
{
  // Don't update anything at the first iteration.
  if (fixed_point_iteration == 0)
    return;

  const auto &fpu = this->param.mesh.adaptation.metric.fixed_point_updates;

  // Update the interface thickness
  {
    const auto &eps_update_data = fpu.chns_interface_thickness;
    if (eps_update_data.enable &&
        fixed_point_iteration % eps_update_data.update_frequency == 0)
    {
      auto &ch = this->param.cahn_hilliard;

      if (fpu.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "-- Updating interface thickness from "
                    << ch.epsilon_interface << " to "
                    << ch.epsilon_interface * eps_update_data.factor
                    << std::endl;

      // Update Cahn-Hilliard parameters
      ch.epsilon_interface *= eps_update_data.factor;

      // FIXME: Update mobility accordingly?

      // Update initial condition
      {
        std::map<std::string, double> new_constants;
        new_constants[eps_update_data.constant_name] = ch.epsilon_interface;
        this->param.initial_conditions.initial_chns_tracer->update_constants(
          new_constants);
      }
    }
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(*this->ordering,
                                               *fe,
                                               *this->fixed_mapping,
                                               *this->moving_mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::setup_assemblers()
{
  assemblers.clear();

  // CHNS assemblers
  Assembly::IncompressibleCHNS::setup_assemblers<dim,
                                                 ScratchData,
                                                 CopyData,
                                                 with_moving_mesh,
                                                 with_enlarged>(
    this->param, *this->ordering, this->coupling_table, assemblers);

  // Elasticity
  if constexpr (with_moving_mesh)
    Assembly::Elasticity::setup_assemblers<dim, ScratchData, CopyData>(
      this->param, *this->ordering, assemblers);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::prepare_timestep()
{
  if (this->param.cahn_hilliard.enable_profile_correction &&
      this->param.mms_param.enable)
  {
    auto source = std::dynamic_pointer_cast<MMSSourceTerm>(this->source_terms);
    AssertThrow(source, ExcInternalError());
    // The manufactured source uses the exact current state, independently of
    // the numerical state and of its Newton or diagnostic perturbations.
    source->set_profile_correction_coefficient(
      compute_profile_correction_coefficient(true));
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
double CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  compute_profile_correction_coefficient(const bool manufactured,
                                         const bool postprocessing) const
{
  const auto &ch = this->param.cahn_hilliard;
  if (!ch.enable_profile_correction)
    return 0.;
  double maximum = ch.mobility;
  if (ch.mobility_model !=
      Parameters::CahnHilliard<dim>::MobilityModel::constant)
  {
    const auto    evaluate = CahnHilliard::get_mobility_evaluation_function(ch);
    const auto    scaling  = CahnHilliard::get_adaptive_mobility_scaling(ch);
    FEValues<dim> values(manufactured ? *this->fixed_mapping :
                                        *this->moving_mapping,
                         *fe,
                         *this->quadrature,
                         update_values | update_gradients |
                           update_quadrature_points);
    const unsigned int               n_q = this->quadrature->size();
    std::vector<double>              phase(n_q);
    std::vector<Tensor<1, dim>>      gradients(n_q), velocities(n_q);
    const FEValuesExtractors::Scalar tracer(this->ordering->phi_lower);
    const FEValuesExtractors::Vector velocity(this->ordering->u_lower);
    std::array<double, 2>            local = {{0., 0.}}, global;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        if (!manufactured)
        {
          const auto &state =
            postprocessing ? *this->present_solution : this->evaluation_point;
          values[tracer].get_function_values(state, phase);
          values[tracer].get_function_gradients(state, gradients);
          values[velocity].get_function_values(state, velocities);
        }
        for (unsigned int q = 0; q < n_q; ++q)
        {
          if (manufactured)
          {
            const auto reference = values.quadrature_point(q);
            Point<dim> point     = reference;
            if constexpr (with_moving_mesh)
              for (unsigned int d = 0; d < dim; ++d)
                point[d] =
                  this->param.mms.exact_mesh_position->value(reference, d);
            phase[q]     = this->param.mms.exact_tracer->value(point);
            gradients[q] = this->param.mms.exact_tracer->gradient(point);
            for (unsigned int d = 0; d < dim; ++d)
              velocities[q][d] =
                this->param.mms.exact_velocity->value(point, d);
          }
          const double mobility =
            evaluate(ch, phase[q], velocities[q], gradients[q], scaling).value;
          if (std::isfinite(mobility))
            local[0] = std::max(local[0], mobility);
          else
            local[1] = 1.;
        }
      }
    // Every rank uses the same amplitude. A non-finite trial produces a
    // non-finite residual so the line search can retry at a smaller step.
    Utilities::MPI::max<double>(make_array_view(local),
                                this->mpi_communicator,
                                make_array_view(global));
    maximum =
      global[1] == 0. ? global[0] : std::numeric_limits<double>::quiet_NaN();
  }
  return ch.profile_correction_strength * 2. *
         (3. / (2. * std::sqrt(2.)) * ch.surface_tension) /
         ch.epsilon_interface * maximum;
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  set_solver_specific_time()
{
  for (auto &[id, bc] : this->param.cahn_hilliard_bc)
    bc.set_time(this->time_handler.current_time);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  create_solver_specific_zero_constraints()
{
  if (this->param.cahn_hilliard.enable_energy_consistent_formulation)
  {
    using Type               = BoundaryConditions::Type;
    unsigned int unsupported = numbers::invalid_unsigned_int;
    // The conservative volume forms omit open-boundary flux integrals.
    // Accept impermeable stationary walls, or full manufactured traces whose
    // test functions vanish. Check mesh faces too, including unspecified IDs.
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
          {
            const auto id        = face->boundary_id();
            const auto fluid     = this->param.fluid_bc.find(id);
            const auto phase     = this->param.cahn_hilliard_bc.find(id);
            bool       supported = false;
            if (fluid != this->param.fluid_bc.end() &&
                phase != this->param.cahn_hilliard_bc.end())
            {
              const bool manufactured =
                this->param.mms_param.enable &&
                fluid->second.type == Type::velocity_mms &&
                phase->second.type == Type::dirichlet_mms;
              bool wall = (fluid->second.type == Type::no_slip ||
                           fluid->second.type == Type::slip) &&
                          phase->second.type == Type::no_flux;
              if constexpr (with_moving_mesh)
              {
                const auto mesh = this->param.pseudosolid_bc.find(id);
                wall = wall && mesh != this->param.pseudosolid_bc.end() &&
                       mesh->second.type == Type::fixed;
              }
              supported = manufactured || wall;
            }
            if (!supported)
              unsupported =
                std::min(unsupported, static_cast<unsigned int>(id));
          }
    unsupported = Utilities::MPI::min(unsupported, this->mpi_communicator);
    AssertThrow(
      unsupported == numbers::invalid_unsigned_int,
      ExcMessage(
        "Unsupported boundary id " + std::to_string(unsupported) +
        " for the energy-consistent formulation. Use no_slip/slip "
        "with phase no_flux (and fixed boundary mesh position in ALE), "
        "or manufactured velocity and phase traces. Open boundaries "
        "and moving physical walls require boundary flux terms."));
  }
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               potential_mask);
      if constexpr (with_enlarged)
        VectorTools::interpolate_boundary_values(
          *this->moving_mapping,
          *this->dof_handler,
          id,
          Functions::ZeroFunction<dim>(this->ordering->n_components),
          this->zero_constraints,
          psi_mask);
    }
    if (bc.type == BoundaryConditions::Type::input_function)
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               tracer_mask);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  create_solver_specific_nonzero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               potential_mask);
      if constexpr (with_enlarged)
        VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                                 *this->dof_handler,
                                                 id,
                                                 *this->exact_solution,
                                                 this->nonzero_constraints,
                                                 psi_mask);
    }
    if (bc.type == BoundaryConditions::Type::input_function)
      VectorTools::interpolate_boundary_values(
        *this->moving_mapping,
        *this->dof_handler,
        id,
        ScalarFunctionFromComponents<dim>(this->ordering->phi_lower,
                                          this->ordering->n_components,
                                          *bc.tracer),
        this->nonzero_constraints,
        tracer_mask);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  set_solver_specific_initial_conditions()
{
  const Function<dim> *tracer_fun =
    this->param.initial_conditions.set_to_mms ?
      this->exact_solution.get() :
      this->param.initial_conditions.initial_chns_tracer.get();

  // Set tracer only
  VectorTools::interpolate(*this->moving_mapping,
                           *this->dof_handler,
                           *tracer_fun,
                           this->newton_update,
                           tracer_mask);

  if constexpr (with_enlarged)
    if (this->param.initial_conditions.set_to_mms &&
        !this->param.cahn_hilliard.use_presolver)
      VectorTools::interpolate(*this->moving_mapping,
                               *this->dof_handler,
                               *this->exact_solution,
                               this->newton_update,
                               psi_mask);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  set_solver_specific_exact_solution()
{
  // Set tracer and potential
  VectorTools::interpolate(*this->moving_mapping,
                           *this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           tracer_mask);
  VectorTools::interpolate(*this->moving_mapping,
                           *this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           potential_mask);
  if constexpr (with_enlarged)
    VectorTools::interpolate(*this->moving_mapping,
                             *this->dof_handler,
                             *this->exact_solution,
                             this->local_evaluation_point,
                             psi_mask);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::create_sparsity_pattern()
{
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);

  const unsigned int n_components   = this->ordering->n_components;
  auto              &coupling_table = this->coupling_table;
  coupling_table = Table<2, DoFTools::Coupling>(n_components, n_components);
  for (unsigned int i = 0; i < n_components; ++i)
    for (unsigned int j = 0; j < n_components; ++j)
    {
      coupling_table[i][j] = DoFTools::none;

      // u couples to all physical variables and the mesh position.
      if (this->ordering->is_velocity(i) && !this->ordering->is_psi(j))
        coupling_table[i][j] = DoFTools::always;

      // p couples to u and x. PSPG also couples p to p, phi and mu through
      // the strong momentum residual.
      if (this->ordering->is_pressure(i) &&
          (this->ordering->is_velocity(j) || this->ordering->is_position(j) ||
           (this->param.stabilization.enable_supg &&
            !this->ordering->is_psi(j))))
        coupling_table[i][j] = DoFTools::always;

      // x couples x,phi,u
      if constexpr (with_moving_mesh)
        if (this->ordering->is_position(i) &&
            (this->ordering->is_position(j) || this->ordering->is_tracer(j) ||
             this->ordering->is_velocity(j)))
          coupling_table[i][j] = DoFTools::always;

      // x also couples to psi: the enlarged moving-mesh forcing compresses the
      // mesh along the widened marker psi.
      if constexpr (with_enlarged)
        if (this->ordering->is_position(i) && this->ordering->is_psi(j))
          coupling_table[i][j] = DoFTools::always;

      // phi couples to u, phi, mu, x
      if (this->ordering->is_tracer(i))
        if (!this->ordering->is_pressure(j) && !this->ordering->is_psi(j))
          coupling_table[i][j] = DoFTools::always;

      // mu couples to phi, mu, u, x
      if (this->ordering->is_potential(i))
        if (!this->ordering->is_pressure(j) && !this->ordering->is_psi(j))
          coupling_table[i][j] = DoFTools::always;

      // psi (enlarged) Helmholtz reconstruction couples to psi, phi, mu, x
      if constexpr (with_enlarged)
        if (this->ordering->is_psi(i) &&
            (this->ordering->is_psi(j) || this->ordering->is_tracer(j) ||
             (this->ordering->is_potential(j) &&
              this->param.cahn_hilliard.psi_mu_correction_factor != 0.) ||
             this->ordering->is_position(j)))
          coupling_table[i][j] = DoFTools::always;
    }

  DoFTools::make_sparsity_pattern(*this->dof_handler,
                                  coupling_table,
                                  dsp,
                                  this->nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             this->locally_owned_dofs,
                                             this->mpi_communicator,
                                             this->locally_relevant_dofs);
  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  scratch_data->profile_correction_coefficient =
    compute_profile_correction_coefficient();

  CopyData copy_data(*fe);

  this->system_matrix = 0;
  this->do_matrix_assembly(*this,
                           this->dof_handler->begin_active(),
                           this->dof_handler->end(),
                           *scratch_data,
                           copy_data,
                           this->param.nonlinear_solver.analytic_jacobian);
  this->system_matrix.compress(VectorOperation::add);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix      = copy_data.local_matrix();
  auto &local_dof_indices = copy_data.dof_indices();
  local_matrix            = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_matrix(scratch_data, copy_data);

  cell->get_dof_indices(local_dof_indices);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                                    copy_data.dof_indices(),
                                                    this->system_matrix);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  compare_analytical_matrix_with_fd()
{
  // Like tau, the global mobility amplitude stays fixed during local finite
  // differences. Only global residual evaluations recompute its maximum.
  scratch_data->profile_correction_coefficient =
    compute_profile_correction_coefficient();
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  scratch_data->profile_correction_coefficient =
    compute_profile_correction_coefficient();

  this->system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  &CHNSSolver::assemble_local_rhs,
                  &CHNSSolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs         = copy_data.local_rhs();
  auto &local_dof_indices = copy_data.dof_indices();
  local_rhs               = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_rhs(scratch_data, copy_data);

  cell->get_dof_indices(local_dof_indices);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                                    copy_data.dof_indices(),
                                                    this->system_rhs);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  compute_solver_specific_errors()
{
  const unsigned int n_active_cells = this->triangulation->n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  const ComponentSelectFunction<dim> tracer_comp_select(
    this->ordering->phi_lower, this->ordering->n_components);
  const ComponentSelectFunction<dim> potential_comp_select(
    this->ordering->mu_lower, this->ordering->n_components);

  this->compute_and_add_errors(*this->moving_mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               tracer_comp_select,
                               "phi");
  this->compute_and_add_errors(*this->moving_mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               potential_comp_select,
                               "mu");
  if constexpr (with_enlarged)
  {
    const ComponentSelectFunction<dim> psi_comp_select(
      this->ordering->psi_lower, this->ordering->n_components);
    this->compute_and_add_errors(*this->moving_mapping,
                                 *this->exact_solution,
                                 cellwise_errors,
                                 psi_comp_select,
                                 "psi");
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  solver_specific_post_processing()
{
  TimerOutput::Scope t(this->computing_timer, "Compute multiphase indicators");
  const auto        &pp = this->param.postprocessing;
  if (pp.chns_conservation.enable || pp.chns_energy.enable ||
      pp.chns_phase_energy.enable || pp.chns_stabilization_energy.enable)
  {
    // Reuse the assembly's physical fields and strong residuals. A private
    // scratch keeps diagnostics from changing the Newton assembly state.
    ScratchData diagnostics(*scratch_data);
    diagnostics.set_keep_tau_constant(false);
    diagnostics.profile_correction_coefficient =
      compute_profile_correction_coefficient(false, true);
    CHNSEnergy::Integrals local, global;
    const bool transient = !this->time_handler.is_steady() &&
                           this->time_handler.current_time_iteration > 0 &&
                           this->time_handler.current_time_iteration !=
                             this->time_handler.time_iteration_at_last_restart;
    std::vector<Tensor<1, dim>> old_gradient(this->quadrature->size()),
      gradient_rate(this->quadrature->size());
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        diagnostics.reinit(cell,
                           *this->present_solution,
                           *this->previous_solutions,
                           *this->source_terms,
                           *this->exact_solution);
        const auto &values = diagnostics.get_current_fe_values();
        std::fill(gradient_rate.begin(), gradient_rate.end(), Tensor<1, dim>());
        if (transient)
        {
          const auto &c = this->time_handler.get_bdf_coefficients();
          for (unsigned int q = 0; q < diagnostics.n_q_points; ++q)
            gradient_rate[q] = c[0] * diagnostics.tracer_gradients[q];
          for (unsigned int k = 1; k < c.size(); ++k)
          {
            values[diagnostics.tracer].get_function_gradients(
              (*this->previous_solutions)[k - 1], old_gradient);
            for (unsigned int q = 0; q < diagnostics.n_q_points; ++q)
              gradient_rate[q] += c[k] * old_gradient[q];
          }
        }
        for (unsigned int q = 0; q < diagnostics.n_q_points; ++q)
          CHNSEnergy::integrate_point<dim>(diagnostics,
                                           q,
                                           gradient_rate[q],
                                           values.quadrature_point(q),
                                           transient ||
                                             this->time_handler.is_steady(),
                                           local);
      }
    Utilities::MPI::sum<double>(make_array_view(local.values),
                                this->mpi_communicator,
                                make_array_view(global.values));
    this->postproc_handler->process_chns_energy(global, this->time_handler);
  }


  this->postproc_handler->compute_multiphase_indicators(*this->ordering,
                                                        *this->dof_handler,
                                                        *this->moving_mapping,
                                                        *this->quadrature,
                                                        *this->present_solution,
                                                        this->time_handler);
}

// Explicit instantiation
template class CHNSSolver<2, false>;
template class CHNSSolver<3, false>;
template class CHNSSolver<2, true>;
template class CHNSSolver<3, true>;
template class CHNSSolver<2, true, true>;
template class CHNSSolver<3, true, true>;
