#include <deal.II/fe/fe_values.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

#include "presolver_test_parameters.h"

// Inspect the initial state through the real CHNS postprocessing hook, then
// let the solver advance through its BDF startup and subsequent time steps.
template <int dim, bool enlarged = false>
class InitialStateProblem : public CHNSSolver<dim, true, enlarged>
{
public:
  InitialStateProblem(const ParameterReader<dim> &param)
    : CHNSSolver<dim, true, enlarged>(param)
  {
    this->pcout.set_condition(false);
  }

  void solver_specific_post_processing() override
  {
    if (this->time_handler.current_time_iteration != 0)
    {
      if (this->time_handler.current_time_iteration == 1)
      {
        double error = 0.;
        for (const auto &[i, position] : initial_positions)
          error =
            std::max(error, std::abs((*this->present_solution)[i] - position));
        AssertThrow(Utilities::MPI::max(error, this->mpi_communicator) < 1e-12,
                    ExcMessage("BDF startup changed the presolved position"));
        checked_startup = true;
      }
      return;
    }
    ++checks;
    if constexpr (enlarged)
      check_reconstruction();
    const auto                      &fe = this->get_fe_system();
    const Quadrature<dim>            nodes(fe.get_unit_support_points());
    FEValues<dim>                    values(*this->fixed_mapping,
                         fe,
                         nodes,
                         update_values | update_quadrature_points);
    const FEValuesExtractors::Vector position(this->ordering->x_lower);
    const FEValuesExtractors::Vector velocity(this->ordering->u_lower);
    const FEValuesExtractors::Scalar phase(this->ordering->phi_lower);
    std::vector<Tensor<1, dim>>      x(nodes.size()), u(nodes.size());
    std::vector<double>              phi(nodes.size());
    double                           displacement = 0., error = 0.;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        values[position].get_function_values(*this->present_solution, x);
        values[velocity].get_function_values(*this->present_solution, u);
        values[phase].get_function_values(*this->present_solution, phi);
        for (unsigned int i = 0; i < nodes.size(); ++i)
        {
          const auto component = fe.system_to_component_index(i).first;
          if (this->ordering->is_tracer(component))
            error = std::max(error,
                             std::abs(phi[i] - std::tanh((x[i][0] - .4) / .2)));
          if (this->ordering->is_velocity(component))
          {
            double expected = .1;
            for (unsigned int d = 0; d < dim; ++d)
              expected *= x[i][d] * (1. - x[i][d]);
            const unsigned int d = component - this->ordering->u_lower;
            error =
              std::max(error, std::abs(u[i][d] - (d == 0 ? expected : 0.)));
          }
          if (this->ordering->is_position(component))
            displacement = std::max(
              displacement,
              (x[i] - Tensor<1, dim>(values.quadrature_point(i))).norm());
        }
        // The reference vertices remain on the original regular grid.
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          for (unsigned int d = 0; d < dim; ++d)
            error = std::max(error,
                             std::abs(3. * cell->vertex(v)[d] -
                                      std::round(3. * cell->vertex(v)[d])));
      }
    for (const auto i : this->locally_owned_dofs)
      if (this->ordering->is_position(
            this->dofs_to_component[this->locally_relevant_dofs
                                      .index_within_set(i)]) ||
          (enlarged && this->ordering->is_psi(
                         this->dofs_to_component[this->locally_relevant_dofs
                                                   .index_within_set(i)])))
        for (const auto &previous : *this->previous_solutions)
          error =
            std::max(error,
                     std::abs(previous[i] - (*this->present_solution)[i]));
    for (const auto i : this->locally_owned_dofs)
      if (this->ordering->is_position(
            this->dofs_to_component[this->locally_relevant_dofs
                                      .index_within_set(i)]) ||
          (enlarged && this->ordering->is_psi(
                         this->dofs_to_component[this->locally_relevant_dofs
                                                   .index_within_set(i)])))
        initial_positions[i] = (*this->present_solution)[i];
    error        = Utilities::MPI::max(error, this->mpi_communicator);
    displacement = Utilities::MPI::max(displacement, this->mpi_communicator);
    AssertThrow(error < 1e-10,
                ExcMessage(
                  "Initial geometry, fields or BDF history are inconsistent"));
    AssertThrow(displacement > 1e-5,
                ExcMessage("The presolver did not compress the mesh"));
    if (this->mpi_rank == 0)
      deallog << dim << "D initial position, fields and BDF histories OK"
              << std::endl;
  }

  void check_reconstruction()
  {
    const auto                          &fe = this->get_fe_system();
    FEValues<dim>                        values(*this->moving_mapping,
                         fe,
                         *this->quadrature,
                         update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);
    const FEValuesExtractors::Scalar     psi(this->ordering->psi_lower);
    std::vector<double>                  marker(values.n_quadrature_points);
    std::vector<Tensor<1, dim>>          gradient(values.n_quadrature_points);
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    LA::ParVectorType                    residual(this->locally_owned_dofs,
                               this->mpi_communicator);
    const auto                          &ch = this->param.cahn_hilliard;
    const double L2 = Utilities::fixed_power<2>(ch.psi_interface_width_factor *
                                                ch.epsilon_interface);
    double       magnitude = 0.;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        cell->get_dof_indices(indices);
        values[psi].get_function_values(*this->present_solution, marker);
        values[psi].get_function_gradients(*this->present_solution, gradient);
        for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
        {
          magnitude = std::max(magnitude, std::abs(marker[q]));
          const double phase =
            this->param.initial_conditions.initial_chns_tracer_callback->value(
              values.quadrature_point(q));
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (this->ordering->is_psi(fe.system_to_component_index(i).first))
              residual[indices[i]] +=
                (values[psi].value(i, q) * (marker[q] - phase) +
                 L2 * values[psi].gradient(i, q) * gradient[q]) *
                values.JxW(q);
        }
      }
    residual.compress(VectorOperation::add);
    AssertThrow(
      residual.linfty_norm() < 2e-9,
      ExcMessage(
        "Injected psi does not satisfy its presolved Helmholtz equation"));
    AssertThrow(Utilities::MPI::max(magnitude, this->mpi_communicator) > 1e-3,
                ExcMessage("Injected psi must be nonzero"));
    if (this->mpi_rank == 0)
      deallog << dim
              << "D injected psi satisfies Helmholtz on the compressed mesh"
              << std::endl;
  }

  unsigned int                              checks          = 0;
  bool                                      checked_startup = false;
  std::map<types::global_dof_index, double> initial_positions;
};

template <int dim, bool enlarged = false>
void check_initial_state()
{
  auto                     param = presolver_test_parameters<dim>();
  if constexpr (enlarged)
  {
    param.cahn_hilliard.mff_enlarged_compression_factor     = 0.5;
    param.cahn_hilliard.psi_interface_width_factor          = 2.;
    param.cahn_hilliard.mff_enlarged_lobe_position_exponent = 0.7;
    param.finite_elements.tracer_degree                     = 2;
    for (auto &[id, bc] : param.pseudosolid_bc)
      bc.type = BoundaryConditions::Type::no_flux;
  }
  InitialStateProblem<dim, enlarged> problem(param);
  problem.run();
  AssertThrow(problem.checks == 1 && problem.checked_startup,
              ExcInternalError());
  if (problem.mpi_rank == 0)
    deallog << dim << "D BDF time steps converged" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  check_initial_state<2>();
  check_initial_state<3>();
  check_initial_state<2, true>();
  check_initial_state<3, true>();
}
