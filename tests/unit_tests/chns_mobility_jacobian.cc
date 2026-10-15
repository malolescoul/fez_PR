#include <assembly/incompressible_chns_assemblers.h>
#include <assembly/elasticity_assemblers.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

namespace
{
  unsigned int failed_directions = 0;

  template <bool moving>
  class State : public Function<2>
  {
  public:
    State(const unsigned int components,
          const unsigned int direction          = numbers::invalid_unsigned_int,
          const bool         pure_phase_at_rest = false)
      : Function<2>(components)
      , direction(direction)
      , pure_phase_at_rest(pure_phase_at_rest)
    {}

    double value(const Point<2> &p, const unsigned int c) const override
    {
      const double x = p[0], y = p[1];
      if (pure_phase_at_rest)
        return c == (moving ? 5 : 3) ? 1. : 0.;
      if (direction != numbers::invalid_unsigned_int)
        return c == direction ?
                 0.3 + 0.2 * x - 0.1 * y + 0.15 * x * y + 0.1 * x * x :
                 0.;
      if (c == 0)
        return 0.4 + 0.3 * x + 0.2 * y * y;
      if (c == 1)
        return -0.2 + 0.2 * y - 0.1 * x * x;
      if (c == 2)
        return 0.2 * x - 0.1 * y;
      if constexpr (moving)
      {
        if (c == 3)
          return x + 0.08 * x * y;
        if (c == 4)
          return y + 0.05 * x * x;
      }
      if (c == (moving ? 5 : 3))
        return -0.3 + 0.5 * x + 0.1 * y * y;
      if (c == (moving ? 6 : 4))
        return 0.1 + 0.3 * x * x + 0.2 * y * y + 0.1 * x * y;
      return 0.1 + 0.2 * x;
    }

  private:
    const unsigned int direction;
    const bool         pure_phase_at_rest;
  };

  template <bool moving, bool enlarged>
  class TestSolver : public CHNSSolver<2, moving, enlarged>
  {
  public:
    using CHNSSolver<2, moving, enlarged>::CHNSSolver;

    void check(const std::string &label)
    {
      this->initialize();
      Triangulation<2> serial_triangulation;
      GridGenerator::hyper_cube(serial_triangulation);
      this->triangulation->create_triangulation(
        TriangulationDescription::Utilities::
          create_description_from_triangulation(serial_triangulation,
                                                this->mpi_communicator));
      this->setup_dofs();
      this->setup_mappings();
      this->create_scratch_data();
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();
      const auto        &fe           = this->dof_handler->get_fe();
      const unsigned int n_components = fe.n_components();
      VectorTools::interpolate(*this->fixed_mapping,
                               *this->dof_handler,
                               State<moving>(n_components,
                                             numbers::invalid_unsigned_int,
                                             this->time_handler.is_steady()),
                               this->local_evaluation_point);
      this->evaluation_point  = this->local_evaluation_point;
      *this->present_solution = this->local_evaluation_point;
      for (auto &previous : *this->previous_solutions)
        previous = this->local_evaluation_point;
      ConditionalOStream quiet(std::cout, false);
      if (!this->time_handler.is_steady())
        this->time_handler.advance(quiet);
      this->set_time();
      this->prepare_timestep();
      // Assembly must refresh the amplitude from the current Newton state,
      // independently of the accepted histories. Each call also represents a
      // possible line-search trial; restoring the state must restore its flux.
      this->assemble_rhs();
      if (this->param.cahn_hilliard.enable_profile_correction)
      {
        const double first = this->scratch_data->profile_correction_coefficient;
        AssertThrow(first > 0.,
                    ExcMessage("Expected positive profile amplitude."));
        (*this->previous_solutions)[0] *= 0.8;
        this->prepare_timestep();
        this->assemble_rhs();
        AssertThrow(
          this->scratch_data->profile_correction_coefficient == first,
          ExcMessage(
            "Current amplitude must not depend on historical fields."));
        LA::ParVectorType original(this->locally_owned_dofs,
                                   this->mpi_communicator);
        LA::ParVectorType perturbation(this->locally_owned_dofs,
                                       this->mpi_communicator);
        original = this->local_evaluation_point;
        for (const unsigned int component :
             {this->ordering->phi_lower, this->ordering->u_lower})
        {
          VectorTools::interpolate(*this->fixed_mapping,
                                   *this->dof_handler,
                                   State<moving>(n_components, component),
                                   perturbation);
          this->local_evaluation_point = original;
          this->local_evaluation_point.add(0.1, perturbation);
          this->evaluation_point = this->local_evaluation_point;
          this->assemble_rhs();
          const double trial =
            this->scratch_data->profile_correction_coefficient;
          const auto model = this->param.cahn_hilliard.mobility_model;
          using Model      = Parameters::CahnHilliard<2>::MobilityModel;
          if ((component == this->ordering->phi_lower &&
               model == Model::degenerate) ||
              (component == this->ordering->u_lower &&
               (model == Model::adaptive_normal_velocity ||
                model == Model::adaptive_velocity)))
            AssertThrow(
              std::abs(trial - first) > 1e-8,
              ExcMessage(
                "Line-search trial did not update the mobility maximum."));
        }
        this->local_evaluation_point = original;
        this->evaluation_point       = original;
        this->assemble_rhs();
        AssertThrow(this->scratch_data->profile_correction_coefficient == first,
                    ExcMessage("Restored iterate must recover its amplitude."));
      }

      using Scratch = NavierStokesScratch::ScratchDataCHNS<2, moving, enlarged>;
      std::vector<
        std::unique_ptr<Assembly::AssemblerBase<Scratch, CopyDataBase<>>>>
        assemblers;
      Assembly::IncompressibleCHNS::
        setup_assemblers<2, Scratch, CopyDataBase<>, moving, enlarged>(
          this->param, *this->ordering, this->coupling_table, assemblers);
      if constexpr (enlarged)
        if (this->param.cahn_hilliard.enable_profile_correction)
          Assembly::Elasticity::setup_assemblers<2, Scratch, CopyDataBase<>>(
            this->param, *this->ordering, assemblers);
      auto       &scratch = *this->scratch_data;
      const auto  cell    = this->dof_handler->begin_active();
      auto        reinit  = [&]() {
        scratch.reinit(cell,
                       this->evaluation_point,
                       *this->previous_solutions,
                       *this->source_terms,
                       *this->exact_solution);
      };
      reinit();
      if (this->param.cahn_hilliard.enable_profile_correction)
      {
        // Independent physical FEValues evaluation also covers non-affine ALE
        // maps. Both the phase profile and mobility use the current geometry.
        FEValues<2> physical(*this->moving_mapping,
                             fe,
                             *this->quadrature,
                             update_values | update_gradients |
                               (scratch.enable_tracer_stabilization ?
                                  update_hessians :
                                  update_default));
        physical.reinit(cell);
        const FEValuesExtractors::Scalar tracer(this->ordering->phi_lower);
        const FEValuesExtractors::Vector velocity(this->ordering->u_lower);
        std::vector<double>              phase(scratch.n_q_points);
        std::vector<Tensor<1, 2>>        gradient(scratch.n_q_points),
          velocities(scratch.n_q_points);
        std::vector<Tensor<2, 2>> hessian(scratch.n_q_points);
        physical[tracer].get_function_values(this->evaluation_point, phase);
        physical[tracer].get_function_gradients(this->evaluation_point,
                                                gradient);
        physical[velocity].get_function_values(this->evaluation_point,
                                               velocities);
        if (scratch.enable_tracer_stabilization)
          physical[tracer].get_function_hessians(this->evaluation_point,
                                                 hessian);
        const auto &ch = this->param.cahn_hilliard;
        const auto  mobility =
          CahnHilliard::get_mobility_evaluation_function(ch);
        const auto scaling = CahnHilliard::get_adaptive_mobility_scaling(ch);
        double     maximum = 0.;
        for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        {
          maximum = std::max(
            maximum,
            mobility(ch, phase[q], velocities[q], gradient[q], scaling).value);
          const auto expected = CahnHilliard::evaluate_profile_correction(
            phase[q],
            gradient[q],
            hessian[q],
            scratch.epsilon,
            scratch.enable_tracer_stabilization);
          const double amplitude = scratch.profile_correction_coefficient;
          AssertThrow(
            (scratch.profile_correction_flux[q] - amplitude * expected.flux)
                .norm() < 1e-11,
            ExcMessage("Profile flux must use the current physical phase."));
          if (scratch.enable_tracer_stabilization)
            AssertThrow(std::abs(scratch.profile_correction_divergence[q] -
                                 amplitude * expected.divergence) < 1e-11,
                        ExcMessage("Incorrect current profile divergence."));
        }
        AssertThrow(
          std::abs(scratch.profile_correction_coefficient -
                   ch.profile_correction_strength * 2. * scratch.sigma_tilde /
                     scratch.epsilon * maximum) < 1e-11,
          ExcMessage("Mobility maximum must use current physical gradients."));
      }
      if (this->time_handler.is_steady())
      {
        for (unsigned int q = 0; q < scratch.n_q_points; ++q)
          AssertThrow(scratch.get_mobility(q) == 0. &&
                        scratch.tau_supg_tracer[q] == 0. &&
                        scratch.get_phase_diffusion(q) == 0.,
                      ExcMessage("Stationary pure phase at rest must have "
                                 "finite zero tracer stabilization."));
        return;
      }
      // As in the approximate Newton matrix, keep tau and the global mobility
      // amplitude fixed while perturbing the local residual.
      scratch.set_keep_tau_constant(true);
      CopyDataBase<> copy(fe);
      copy.cell_is_at_boundary = cell->at_boundary();
      copy.local_matrix()      = 0.;
      for (const auto &assembler : assemblers)
        assembler->assemble_matrix(scratch, copy);
      const FullMatrix<double>             matrix(copy.local_matrix());
      std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      LA::ParVectorType base(this->locally_owned_dofs, this->mpi_communicator);
      LA::ParVectorType direction(this->locally_owned_dofs,
                                  this->mpi_communicator);
      base = this->local_evaluation_point;

      double worst_error = 0.;
      // Separate component directions prevent the large momentum/time terms
      // from hiding a missing tracer or pressure coupling.
      for (unsigned int component = 0; component < n_components; ++component)
      {
        VectorTools::interpolate(*this->fixed_mapping,
                                 *this->dof_handler,
                                 State<moving>(n_components, component),
                                 direction);
        Vector<double> local_direction(indices.size()),
          analytic(indices.size());
        for (unsigned int j = 0; j < indices.size(); ++j)
          local_direction[j] = direction[indices[j]];
        matrix.vmult(analytic, local_direction);
        double             best_error = std::numeric_limits<double>::max();
        std::ostringstream diagnostics;
        diagnostics << std::scientific << std::setprecision(8);
        for (const double h : {2e-5, 5e-6, 1e-6})
        {
          auto rhs = [&](const double step) {
            this->local_evaluation_point = base;
            this->local_evaluation_point.add(step, direction);
            this->evaluation_point = this->local_evaluation_point;
            reinit();
            copy.local_rhs() = 0.;
            for (const auto &assembler : assemblers)
              assembler->assemble_rhs(scratch, copy);
            return Vector<double>(copy.local_rhs());
          };
          Vector<double> numerical = rhs(h);
          numerical -= rhs(-h);
          numerical *= -0.5 / h;
          double maximum_error = 0.;
          for (unsigned int row_component = 0; row_component < n_components;
               ++row_component)
          {
            double error_sq = 0., scale_sq = 0., numerical_sq = 0.;
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (fe.system_to_component_index(i).first == row_component)
              {
                error_sq +=
                  Utilities::fixed_power<2>(numerical[i] - analytic[i]);
                scale_sq += Utilities::fixed_power<2>(analytic[i]);
                numerical_sq += Utilities::fixed_power<2>(numerical[i]);
              }
            const double row_error =
              std::sqrt(error_sq) / std::max(1., std::sqrt(scale_sq));
            AssertThrow(std::isfinite(error_sq) && std::isfinite(scale_sq) &&
                          std::isfinite(numerical_sq) &&
                          std::isfinite(row_error),
                        ExcMessage(
                          "Non-finite Jacobian component comparison."));
            maximum_error = std::max(maximum_error, row_error);
            if (row_error >= 2e-7)
              diagnostics << "  h=" << h << " row=" << row_component
                          << " error_norm=" << std::sqrt(error_sq)
                          << " analytic_norm=" << std::sqrt(scale_sq)
                          << " numerical_norm=" << std::sqrt(numerical_sq)
                          << " relative_error=" << row_error << '\n';
          }
          best_error = std::min(best_error, maximum_error);
        }
        worst_error = std::max(worst_error, best_error);
        if (best_error >= 2e-7)
        {
          ++failed_directions;
          std::cerr << label << ": column component=" << component
                    << " best directional error=" << best_error << '\n'
                    << diagnostics.str();
        }
      }
      if (this->param.cahn_hilliard.enable_profile_correction)
        std::cerr << label << " maximum directional error " << worst_error
                  << '\n';
    }
  };

  template <bool moving, bool enlarged = false>
  void run_case(const std::string &mobility_model,
                const bool         momentum_supg,
                const bool         tracer_supg,
                const bool         stationary         = false,
                const bool         profile_correction = false)
  {
    Parameters::BoundaryConditionsData bc;
    ParameterHandler                   prm;
    ParameterReader<2>                 param(bc);
    param.declare(prm);
    std::ostringstream input;
    input << R"(
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
end
subsection Time integration
  set scheme = BDF1
  set dt = 0.2
  set t_end = 1
  set verbosity = quiet
end
subsection FiniteElements
  set use quads = true
  set Velocity degree = 2
  set Pressure degree = 1
  set Mesh position degree = 2
  set Tracer degree = 2
  set Potential degree = 2
end
subsection Physical properties
  set number of fluids = 2
  subsection Fluid 0
    set density = 3
    set kinematic viscosity = 0.05
  end
  subsection Fluid 1
    set density = 1
    set kinematic viscosity = 0.05
  end
  set number of pseudosolids = 1
  subsection Pseudosolid 0
    subsection lame lambda
      set Function expression = 1
    end
    subsection lame mu
      set Function expression = 1
    end
  end
end
subsection Cahn Hilliard
  set surface tension = 1
  set interface thickness = 0.5
  set enable tracer limiter = false
)";
    input
      << "  subsection mobility\n    set model = " << mobility_model << '\n'
      << "    set scaling factor = 2\n    set mobility regularization = 0.15\n"
      << "    set velocity regularization = 0.15\n"
      << "    set Function expression = 0.2*(1-x*x)^2\n  end\nend\n"
      << "subsection Stabilization\n  set enable supg = "
      << (momentum_supg ? "true" : "false")
      << "\n  set enable tracer supg = " << (tracer_supg ? "true" : "false")
      << "\nend\n";
    auto text = input.str();
    if (profile_correction)
      text.replace(text.find("set scheme = BDF1"),
                   17,
                   "set scheme = BDF2\n  set bdf start method = BDF1");
    if (stationary)
    {
      text.replace(text.find("set scheme = BDF1"),
                   17,
                   "set scheme = stationary");
      // Force exactly zero mobility; interpolating phi=1 can leave roundoff
      // in (1-phi^2)^2 and would not exercise the zero-denominator branch.
      const std::string expression = "0.2*(1-x*x)^2";
      text.replace(text.find(expression), expression.size(), "0");
    }
    if constexpr (enlarged)
      if (profile_correction)
        text += R"(
subsection Cahn Hilliard
  set psi interface width factor = 2
  set psi mu correction factor = 0.3
  set mff source term = chns form
  set mff physics compression factor = 0.5
  set mff enlarged compression factor = 0.5
  set mff transport factor = 0.1
  set mff regularization gamma = 0.5
  set mff enlarged lobe position exponent = 0.7
end
)";
    std::istringstream parameters(text);
    prm.parse_input(parameters);
    param.read(prm);
    param.cahn_hilliard.enable_profile_correction = profile_correction;
    param.bc_data.fix_pressure_constant      = false;
    param.bc_data.enforce_zero_mean_pressure = false;
    const std::string label = mobility_model + (enlarged ? "/enlarged" : "") +
                              (moving ? "/ALE" : "/fixed") +
                              (momentum_supg ? "/momentum-SUPG" : "") +
                              (tracer_supg ? "/tracer-SUPG" : "");
    TestSolver<moving, enlarged>(param).check(
      label + (profile_correction ? "/profile-correction" : ""));
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  for (const std::string model :
       {"degenerate", "Adaptive normal velocity", "Adaptive velocity"})
    for (const bool stabilized : {false, true})
    {
      run_case<false>(model, stabilized, stabilized);
#if !DEAL_II_VERSION_GTE(9, 9, 0)
      // Earlier deal.II versions cannot transform MappingFEField Hessians.
      if (!stabilized)
#endif
      {
        run_case<true>(model, stabilized, stabilized);
        run_case<true, true>(model, stabilized, stabilized);
      }
    }
  for (const std::string model : {"constant",
                                  "degenerate",
                                  "Adaptive normal velocity",
                                  "Adaptive velocity"})
    for (const bool stabilized : {false, true})
    {
      run_case<false>(model, stabilized, stabilized, false, true);
#if !DEAL_II_VERSION_GTE(9, 9, 0)
      if (!stabilized)
#endif
      {
        run_case<true>(model, stabilized, stabilized, false, true);
        run_case<true, true>(model, stabilized, stabilized, false, true);
      }
    }
  run_case<false>("degenerate", false, true, true);
  AssertThrow(failed_directions == 0,
              ExcMessage(std::to_string(failed_directions) +
                         " mobility component directions failed"));
  initlog();
  deallog << "Mobility Jacobians: fixed, ALE and enlarged, with and without "
             "stabilization, and fixed/ALE/enlarged profile correction OK"
          << std::endl;
}
