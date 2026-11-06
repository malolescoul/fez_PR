#ifndef CHNS_CAPILLARY_PROJECTION_H
#define CHNS_CAPILLARY_PROJECTION_H

#include <chns_surface_tension.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <types.h>

namespace CHNSSurfaceTension
{
  /** Continuous linear, row-sum mass-lumped weak recovery of div(T).
   * Includes the physical boundary traction and interelement stress jumps.
   * No velocity boundary constraints are imposed on this diagnostic field.
   * Hanging-node constraints preserve partition of unity and the resultant.
   */
  template <int dim>
  class WeakForceProjection
  {
  public:
    explicit WeakForceProjection(const Triangulation<dim> &tria)
      : dofs(tria)
    {}

    template <typename VectorType>
    std::pair<Tensor<1, dim>, double>
    compute(const DoFHandler<dim> &source,
            const Mapping<dim>    &mapping,
            const Quadrature<dim> &quadrature,
            const VectorType      &solution,
            const unsigned int     phase_component,
            const double           sigma)
    {
      dofs.clear();
      const bool simplex = source.get_fe().reference_cell().is_simplex();
      if (simplex)
        fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(1), dim);
      else
        fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(1), dim);
      dofs.distribute_dofs(*fe);
      const auto comm     = source.get_mpi_communicator();
      const auto owned    = dofs.locally_owned_dofs();
      const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dofs, constraints);
      constraints.close();
      LA::ParVectorType rhs, mass, result;
      rhs.reinit(owned, comm);
      mass.reinit(owned, comm);
      result.reinit(owned, comm);
      field.reinit(owned, relevant, comm);
      FEValues<dim> src(mapping, source.get_fe(), quadrature, update_gradients);
      FEValues<dim> dst(mapping,
                        *fe,
                        quadrature,
                        update_values | update_gradients | update_JxW_values);
      std::unique_ptr<Quadrature<dim - 1>> fq;
      const unsigned int                   order = source.get_fe().degree + 2;
      if (simplex)
        fq = std::make_unique<QGaussSimplex<dim - 1>>(order);
      else
        fq = std::make_unique<QGauss<dim - 1>>(order);
      FEFaceValues<dim> sf(mapping, source.get_fe(), *fq, update_gradients);
      FEFaceValues<dim> df(mapping,
                           *fe,
                           *fq,
                           update_values | update_normal_vectors |
                             update_JxW_values);
      const FEValuesExtractors::Scalar phase(phase_component);
      const FEValuesExtractors::Vector vector(0);
      std::vector<Tensor<1, dim>>      gradients(quadrature.size()),
        face_gradients(fq->size());
      Vector<double> local_rhs(fe->n_dofs_per_cell()),
        local_mass(fe->n_dofs_per_cell());
      std::vector<types::global_dof_index> indices(fe->n_dofs_per_cell());
      Tensor<1, dim>                       local_force;
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          const auto scell = cell->as_dof_handler_iterator(source);
          src.reinit(scell);
          dst.reinit(cell);
          src[phase].get_function_gradients(solution, gradients);
          local_rhs  = 0.;
          local_mass = 0.;
          for (unsigned q = 0; q < quadrature.size(); ++q)
          {
            const auto T = stress(gradients[q], sigma);
            for (unsigned i = 0; i < fe->n_dofs_per_cell(); ++i)
            {
              const unsigned c = fe->system_to_component_index(i).first;
              local_rhs(i) -=
                scalar_product(T, dst[vector].gradient(i, q)) * dst.JxW(q);
              local_mass(i) += dst.shape_value_component(i, q, c) * dst.JxW(q);
            }
          }
          for (unsigned f = 0; f < cell->n_faces(); ++f)
            if (cell->face(f)->at_boundary())
            {
              sf.reinit(scell, f);
              df.reinit(cell, f);
              sf[phase].get_function_gradients(solution, face_gradients);
              for (unsigned q = 0; q < fq->size(); ++q)
              {
                const auto traction =
                  stress(face_gradients[q], sigma) * df.normal_vector(q);
                local_force += traction * df.JxW(q);
                for (unsigned i = 0; i < fe->n_dofs_per_cell(); ++i)
                  local_rhs(i) +=
                    (df[vector].value(i, q) * traction) * df.JxW(q);
              }
            }
          cell->get_dof_indices(indices);
          constraints.distribute_local_to_global(local_rhs, indices, rhs);
          constraints.distribute_local_to_global(local_mass, indices, mass);
        }
      rhs.compress(VectorOperation::add);
      mass.compress(VectorOperation::add);
      for (const auto i : owned)
        result[i] = mass[i] > 0. ? rhs[i] / mass[i] : 0.;
      result.compress(VectorOperation::insert);
      constraints.distribute(result);
      field                                       = result;
      double                      local_magnitude = 0.;
      std::vector<Tensor<1, dim>> values(quadrature.size());
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          dst.reinit(cell);
          dst[vector].get_function_values(field, values);
          for (unsigned q = 0; q < quadrature.size(); ++q)
            local_magnitude += values[q].norm() * dst.JxW(q);
        }
      Tensor<1, dim> resultant;
      for (unsigned d = 0; d < dim; ++d)
        resultant[d] = Utilities::MPI::sum(local_force[d], comm);
      return {resultant, Utilities::MPI::sum(local_magnitude, comm)};
    }

    template <typename DataOutType>
    void add_data(DataOutType &out) const
    {
      out.add_data_vector(
        dofs,
        field,
        std::vector<std::string>(dim, "surface_tension_force"),
        std::vector<DataComponentInterpretation::DataComponentInterpretation>(
          dim, DataComponentInterpretation::component_is_part_of_vector));
    }

  private:
    std::unique_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dofs;
    LA::ParVectorType                   field;
  };
} // namespace CHNSSurfaceTension
#endif
