#include <deal.II/dofs/dof_tools.h>
#include <incompressible_chns_solver.h>
#include <mesh.h>

#include <sstream>

#include "../tests.h"

// Adapted from PR101. These parameters deliberately require no presolver.
template <int dim>
ParameterReader<dim> parameters(const bool deformed_initial_mesh = false,
                                const bool translated_boundary   = false)
{
  Parameters::BoundaryConditionsData boundaries{};
  boundaries.n_fluid_bc           = boundaries.n_pseudosolid_bc =
    boundaries.n_cahn_hilliard_bc = 2 * dim;
  ParameterReader<dim> param(boundaries);
  ParameterHandler     prm;
  param.declare(prm);
  prm.parse_input_from_string(R"(
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
  set output directory = ./
end
subsection Mesh
  set verbosity = quiet
  set dealii preset mesh = rectangle
  set refinement level = 1
end
subsection Time integration
  set verbosity = quiet
  set scheme = stationary
end
subsection FiniteElements
  set use quads = true
  set Velocity degree = 2
  set Pressure degree = 1
  set Mesh position degree = 1
  set Tracer degree = 1
  set Potential degree = 1
end
subsection Physical properties
  set number of fluids = 2
  subsection Fluid 0
    set density = 1
    set kinematic viscosity = 1
  end
  subsection Fluid 1
    set density = 1
    set kinematic viscosity = 1
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
subsection Fluid boundary conditions
  set fix pressure constant = false
end
subsection Cahn Hilliard
  set alpha = 0
  set beta = 0
end
)");
  prm.enter_subsection("Dimension");
  prm.set("dimension", std::to_string(dim));
  prm.leave_subsection();
  prm.enter_subsection("Mesh");
  prm.set("dealii mesh parameters",
          dim == 2 ? "2,2:0,0:1,1:true" : "2,2,2:0,0,0:1,1,1:true");
  prm.leave_subsection();
  for (const std::string section : {"Fluid boundary conditions",
                                    "Pseudosolid boundary conditions",
                                    "CahnHilliard boundary conditions"})
  {
    prm.enter_subsection(section);
    prm.set("number", std::to_string(2 * dim));
    for (unsigned int b = 0; b < 2 * dim; ++b)
    {
      prm.enter_subsection("boundary " + std::to_string(b));
      prm.set("id", std::to_string(b));
      prm.set("name",
              std::string(1, "xyz"[b / 2]) + (b % 2 == 0 ? "_min" : "_max"));
      prm.set("type",
              section == "Fluid boundary conditions"       ? "no_slip" :
              section == "Pseudosolid boundary conditions" ? "no_flux" :
                                                             "input_function");
      if (section == "CahnHilliard boundary conditions")
      {
        prm.enter_subsection("tracer");
        prm.set("Function expression", dim == 2 ? "x+2*y+t" : "x+2*y+3*z+t");
        prm.leave_subsection();
      }
      if (translated_boundary && section == "Pseudosolid boundary conditions")
      {
        prm.set("type", "input_function");
        prm.enter_subsection("x");
        prm.set("Function expression", "x+0.05");
        prm.leave_subsection();
        prm.enter_subsection("y");
        prm.set("Function expression", "y");
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
  if (deformed_initial_mesh)
    prm.parse_input_from_string(R"(
subsection Manufactured solution
  set enable = true
end
subsection Initial conditions
  set to mms = true
end
subsection Exact solution
  subsection exact mesh displacement
    set Function expression = x+0.1*x*(1-x);y
  end
end
)");
  param.read(prm);
  return param;
}

template <int dim, bool moving>
class BoundaryProblem : public CHNSSolver<dim, moving>
{
public:
  BoundaryProblem(const ParameterReader<dim> &param)
    : CHNSSolver<dim, moving>(param)
  {
    this->pcout.set_condition(false);
    this->reset();
    this->initialize();
    this->set_interval_data(0);
    MeshTools::read_mesh(*this->triangulation, this->param);
    this->setup_dofs();
    this->setup_mappings();
  }

  void check_initial_geometry(const bool translated_boundary = false)
  {
    const auto points =
      DoFTools::map_dofs_to_support_points(*this->fixed_mapping,
                                           *this->dof_handler);
    this->time_handler.current_time = .375;
    this->set_time();
    this->create_zero_constraints();
    this->create_nonzero_constraints();
    this->set_initial_conditions(false);
    for (const auto i : this->locally_owned_dofs)
      if (this->ordering->is_tracer(
            this->dofs_to_component[this->locally_relevant_dofs
                                      .index_within_set(i)]) &&
          this->nonzero_constraints.is_constrained(i))
      {
        const auto  &p = points.at(i);
        const double expected =
          p[0] + (translated_boundary ? .05 : .1 * p[0] * (1. - p[0])) +
          2. * p[1] + .375;
        AssertThrow(std::abs(this->evaluation_point[i] - expected) < 1e-11,
                    ExcMessage("Initial tracer uses the reference geometry"));
      }
    if (this->mpi_rank == 0)
      deallog << "Initial ALE tracer geometry OK" << std::endl;
  }

  void check()
  {
    const auto points =
      DoFTools::map_dofs_to_support_points(*this->fixed_mapping,
                                           *this->dof_handler);
    for (const double time : {.375, .625})
    {
      this->time_handler.current_time = time;
      this->set_time();
      for (const double alpha : {1., .5, .125})
      {
        for (const auto i : this->locally_owned_dofs)
        {
          const auto c     = this->dofs_to_component[this->locally_relevant_dofs
                                                   .index_within_set(i)];
          double     value = .123;
          if constexpr (moving)
            if (this->ordering->is_position(c))
            {
              const double x = points.at(i)[c - this->ordering->x_lower];
              value          = x + .1 * alpha * x * (1. - x);
            }
          this->local_evaluation_point[i] = value;
        }
        this->local_evaluation_point.compress(VectorOperation::insert);
        // Keep the mapping on the preceding configuration until the production
        // operation refreshes it, exactly as for a Newton trial.
        this->create_zero_constraints();
        this->create_nonzero_constraints();
        this->distribute_nonzero_constraints();
        this->evaluation_point = this->local_evaluation_point;
        unsigned int checked   = 0;
        for (const auto i : this->locally_owned_dofs)
        {
          const auto  c = this->dofs_to_component[this->locally_relevant_dofs
                                                   .index_within_set(i)];
          const auto &p = points.at(i);
          bool        boundary   = false;
          double      prescribed = time;
          for (unsigned int d = 0; d < dim; ++d)
          {
            boundary = boundary || p[d] < 1e-12 || p[d] > 1. - 1e-12;
            const double x =
              p[d] + (moving ? .1 * alpha * p[d] * (1. - p[d]) : 0.);
            prescribed += (d + 1.) * x;
          }
          if (this->ordering->is_tracer(c))
          {
            AssertThrow(this->nonzero_constraints.is_constrained(i) == boundary,
                        ExcMessage("Incorrect tracer constraint selection"));
            const double expected = boundary ? prescribed : .123;
            AssertThrow(
              std::abs(this->local_evaluation_point[i] - expected) < 1e-11,
              ExcMessage("Prescribed tracer uses stale coordinates or time"));
            if (boundary)
            {
              ++checked;
              AssertThrow(this->zero_constraints.is_constrained(i) &&
                            this->zero_constraints.get_inhomogeneity(i) == 0.,
                          ExcMessage("Nonzero homogeneous tracer correction"));
            }
          }
          if (c == this->ordering->mu_lower || c == this->ordering->p_lower)
            AssertThrow(!this->nonzero_constraints.is_constrained(i) &&
                          std::abs(this->local_evaluation_point[i] - .123) <
                            1e-12,
                        ExcMessage("An unrelated scalar was constrained"));
        }
        AssertThrow(Utilities::MPI::sum(checked, this->mpi_communicator) > 0,
                    ExcInternalError());
      }
    }
    if (this->mpi_rank == 0)
      deallog << dim << "D " << (moving ? "moving" : "fixed")
              << " tracer values, time and free fields OK" << std::endl;
  }
};

// Exercise the actual Newton driver with CHNS geometry and boundary
// constraints. Only the residual and linear update are controlled, as in
// newton_safeguards: the full step inverts cells, and half of it reaches the
// prescribed root.
class NewtonBoundaryProblem : public BoundaryProblem<2, true>
{
public:
  static ParameterReader<2> make_parameters()
  {
    auto param                                = parameters<2>();
    param.nonlinear_solver.verbosity          = Parameters::Verbosity::quiet;
    param.nonlinear_solver.tolerance          = 1e-10;
    param.nonlinear_solver.max_iterations     = 2;
    param.nonlinear_solver.line_search.enable = true;
    param.nonlinear_solver.line_search.memory = 1;
    param.nonlinear_solver.line_search.max_backtrack_steps = 3;
    param.linear_solver.at(SolverInfo::SolverType::main_physics).method =
      Parameters::LinearSolver::Method::gmres;
    // Rebuilding this condition requires boundary normals on the ALE mapping.
    param.fluid_bc.at(2).type       = BoundaryConditions::Type::slip;
    param.pseudosolid_bc.at(2).type = BoundaryConditions::Type::none;
    return param;
  }

  NewtonBoundaryProblem(const bool reject_all, const double full_step = -2.5)
    : BoundaryProblem<2, true>(make_parameters())
    , reject_all(reject_all)
    , full_step(full_step)
  {
    points = DoFTools::map_dofs_to_support_points(*fixed_mapping, *dof_handler);
    time_handler.current_time = .375;
    set_time();
    create_zero_constraints();
    create_nonzero_constraints();
    distribute_nonzero_constraints();
    evaluation_point  = local_evaluation_point;
    *present_solution = evaluation_point;
  }

  bool evaluation_point_is_valid() const override
  {
    const bool valid = CHNSSolver<2, true>::evaluation_point_is_valid();
    invalid_seen     = invalid_seen || !valid;
    return valid;
  }

  double amplitude() const
  {
    double       sum   = 0.;
    unsigned int count = 0;
    for (const auto i : locally_owned_dofs)
      if (dofs_to_component[locally_relevant_dofs.index_within_set(i)] ==
            ordering->x_lower &&
          std::abs(points.at(i)[0] - .5) < 1e-12)
      {
        sum += (evaluation_point[i] - .5) / .25;
        ++count;
      }
    const auto total = Utilities::MPI::sum(count, mpi_communicator);
    AssertThrow(total > 0, ExcInternalError());
    return Utilities::MPI::sum(sum, mpi_communicator) / total;
  }

  void check_tracer() const
  {
    const double a = amplitude();
    for (const auto i : locally_owned_dofs)
      if (ordering->is_tracer(
            dofs_to_component[locally_relevant_dofs.index_within_set(i)]))
      {
        const auto &p = points.at(i);
        if (p[0] < 1e-12 || p[0] > 1. - 1e-12 || p[1] < 1e-12 ||
            p[1] > 1. - 1e-12)
          AssertThrow(std::abs(evaluation_point[i] -
                               (p[0] + a * p[0] * (1. - p[0]) +
                                2. * (p[1] + shear() * a * p[0] * (1. - p[0]) *
                                               (1. - p[1])) +
                                .375)) < 1e-11,
                      ExcMessage(
                        "Newton tracer values use the wrong geometry"));
      }
  }

  double shear() const { return full_step < -2. ? .05 : 0.; }

  void assemble_matrix() override {}
  void assemble_rhs() override
  {
    AssertThrow((CHNSSolver<2, true>::evaluation_point_is_valid()),
                ExcMessage("Invalid trial reached RHS assembly"));
    check_tracer();
    const double residual =
      reject_all ? 1. : (amplitude() - .5 * full_step) / (-.5 * full_step);
    system_rhs =
      residual / std::sqrt(static_cast<double>(dof_handler->n_dofs()));
  }
  void solve_linear_system_iterative() override
  {
    newton_update = 0.;
    for (const auto i : locally_owned_dofs)
      if (dofs_to_component[locally_relevant_dofs.index_within_set(i)] ==
          ordering->x_lower)
      {
        const double x   = points.at(i)[0];
        newton_update[i] = full_step * x * (1. - x);
      }
      else if (dofs_to_component[locally_relevant_dofs.index_within_set(i)] ==
               ordering->x_lower + 1)
      {
        const auto &p = points.at(i);
        newton_update[i] =
          shear() * full_step * p[0] * (1. - p[0]) * (1. - p[1]);
      }
    newton_update.compress(VectorOperation::insert);
  }

  void check_newton()
  {
    bool failed = false;
    try
    {
      solve_nonlinear_problem(time_handler);
    }
    catch (const std::runtime_error &error)
    {
      if (std::string(error.what()).find("line search failed") ==
          std::string::npos)
        throw;
      failed = true;
    }
    AssertThrow(failed == reject_all && invalid_seen,
                ExcMessage("Unexpected Newton acceptance/rejection"));
    AssertThrow(
      std::abs(amplitude() - (reject_all ? 0. : .5 * full_step)) < 1e-11,
      ExcMessage("Newton did not restore/accept the expected geometry"));
    check_tracer();
    if (reject_all)
    {
      // The real retry path updates BC before assembling its first residual.
      update_boundary_conditions();
      std::ostringstream actual, expected;
      zero_constraints.print(actual);
      create_zero_constraints();
      zero_constraints.print(expected);
      AssertThrow(Utilities::MPI::min(actual.str() == expected.str() ? 1 : 0,
                                      mpi_communicator) == 1,
                  ExcMessage("Retry retains rejected slip constraints"));
    }
    // Reapplying constraints after failure must not reintroduce the last
    // rejected geometry's tracer values.
    local_evaluation_point = evaluation_point;
    distribute_nonzero_constraints();
    evaluation_point = local_evaluation_point;
    check_tracer();
    if (mpi_rank == 0)
      deallog << "Newton tracer "
              << (reject_all ? "restoration" : "backtracking") << " OK"
              << std::endl;
  }

private:
  const bool                                  reject_all;
  const double                                full_step;
  mutable bool                                invalid_seen = false;
  std::map<types::global_dof_index, Point<2>> points;
};

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  deallog.depth_console(0);
  mpi_initlog();
  try
  {
    BoundaryProblem<2, false>(parameters<2>()).check();
    BoundaryProblem<2, true>(parameters<2>()).check();
    BoundaryProblem<3, false>(parameters<3>()).check();
    BoundaryProblem<3, true>(parameters<3>()).check();
    BoundaryProblem<2, true>(parameters<2>(true)).check_initial_geometry();
    BoundaryProblem<2, true>(parameters<2>(false, true))
      .check_initial_geometry(true);
    NewtonBoundaryProblem(false).check_newton();
    NewtonBoundaryProblem(true).check_newton();
    // On the 2x2 mesh, a = -2 collapses the first boundary edge exactly.
    NewtonBoundaryProblem(false, -2.).check_newton();
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
