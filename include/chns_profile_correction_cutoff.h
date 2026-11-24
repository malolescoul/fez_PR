#ifndef CHNS_PROFILE_CORRECTION_CUTOFF_H
#define CHNS_PROFILE_CORRECTION_CUTOFF_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/solution_transfer.h>
#include <types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

/**
 * Interaction cutoff of the profile correction.
 *
 * The profile correction drives |grad(phi)| towards the profile of a planar,
 * isolated interface. Where two interfaces overlap (liquid or gas filaments,
 * notches) or where an interface folds below the scale epsilon, this target no
 * longer exists and the correction turns anti-diffusive. The cutoff chi in
 * [0,1] detects these zones from phi alone, and the correction coefficient
 * becomes K (1 - chi). See docs/correction-adaptative for the construction.
 *
 * Optionally, the coefficient is also scaled by a ratio r >= 1 so that the
 * correction holds the stretching of the interface by the flow: a normal
 * strain n.grad(u).n spreads the isolines, and the correction restores them
 * at the rate 2 K gain(phi) / eps^2. The coefficient is then K r (1 - chi).
 */
namespace CahnHilliard
{
  using namespace dealii;

  /**
   * Smooth ramp s(x) = 3x^2 - 2x^3 on [0,1], equal to 0 below and 1 above.
   */
  inline double cutoff_ramp(const double x)
  {
    const double y = std::clamp(x, 0., 1.);
    return y * y * (3. - 2. * y);
  }

  /**
   * User parameters of the cutoff. The defaults reproduce the first version:
   * curvature ramp from 1 to 2 and no relaxation in time.
   */
  struct InteractionCutoffParameters
  {
    // Start of the |kappa eps| ramp, complete at twice this value.
    double kappa_start = 1.;
    // Relaxation times of chi towards its detected value, in time units. Zero
    // makes chi follow the detection instantaneously.
    double rise_time  = 0.;
    double decay_time = 0.;
    // Detection of the interactions; chi stays zero when disabled.
    bool cutoff = true;
    // Scaling by the normal strain of the flow, measured where |phi| is below
    // strain_band. strain_reference is the strain held by the unscaled
    // correction, so that r = clamp(strain_safety * strain / strain_reference,
    // 1, max_strain_ratio).
    bool   strain_scaling   = false;
    double strain_band      = 0.9999;
    double strain_reference = 1.;
    double strain_safety    = 2.;
    double max_strain_ratio = 10.;
  };

  /**
   * Pointwise indicators and raw cutoff weights. The weights are split by the
   * phase that is pinched: phi > 0 (positive) or phi < 0 (negative).
   */
  struct InteractionIndicator
  {
    double lambda        = 0.;
    double kappa_epsilon = 0.;
    // Weight of the reliable core |phi| < 0.95, and whether kappa*eps is
    // defined (normal well defined, away from midlines).
    double core          = 0.;
    bool   kappa_defined = false;
    double positive      = 0.;
    double negative      = 0.;
  };

  /**
   * Raw cutoff weights of the two phases from Lambda, kappa*eps and the core
   * weight. kappa*eps > 0 means that the centre of curvature lies in phi > 0,
   * where the isocontours converge.
   */
  inline void set_interaction_weights(InteractionIndicator &indicator,
                                      const double          kappa_start = 1.)
  {
    // Thresholds of the cutoff, see the table in docs/correction-adaptative.
    constexpr double lambda_start = 0.5, lambda_full = 1.;
    const double     kappa_full = 2. * kappa_start;
    const double     c_lambda =
      cutoff_ramp((std::abs(indicator.lambda) - lambda_start) /
                  (lambda_full - lambda_start));
    const double c_kappa =
      indicator.kappa_defined ?
        cutoff_ramp((std::abs(indicator.kappa_epsilon) - kappa_start) /
                    (kappa_full - kappa_start)) :
        0.;
    indicator.positive =
      indicator.core * std::max(indicator.lambda < 0. ? c_lambda : 0.,
                                indicator.kappa_epsilon > 0. ? c_kappa : 0.);
    indicator.negative =
      indicator.core * std::max(indicator.lambda > 0. ? c_lambda : 0.,
                                indicator.kappa_epsilon < 0. ? c_kappa : 0.);
  }

  // Start of the reliable core of the interface.
  constexpr double interaction_core_outer = 0.95;
  // Bound of |phi| in psi = atanh(phi). It lies well outside the core: psi
  // stops being linear across a tanh profile at this bound, which must not be
  // read as a curvature by the cells of the core.
  constexpr double interaction_psi_bound = 0.995;

  /**
   * Gradient of psi = atanh(phi), with phi clamped to the bound. It is
   * constant on a planar tanh profile, whatever its width.
   */
  template <int dim>
  Tensor<1, dim> psi_gradient(const double phi, const Tensor<1, dim> &gradient)
  {
    const double p =
      std::clamp(phi, -interaction_psi_bound, interaction_psi_bound);
    return gradient / (1. - p * p);
  }

  /**
   * Hessian of psi = atanh(phi) from the derivatives of phi at one point.
   */
  template <int dim>
  Tensor<2, dim> psi_hessian(const double          phi,
                             const Tensor<1, dim> &gradient,
                             const Tensor<2, dim> &hessian)
  {
    const double p =
      std::clamp(phi, -interaction_psi_bound, interaction_psi_bound);
    const double w = 1. - p * p;
    return hessian / w + 2. * p / (w * w) * outer_product(gradient, gradient);
  }

  /**
   * Evaluate, at one point, the two detectors of the cutoff from phi, its
   * gradient and the Hessian of psi = atanh(phi):
   *
   * - Lambda = 2 eps^2 d^2(psi)/dn^2 vanishes on any tanh profile built on a
   *   distance, whatever its width or curvature, and is about -(3+phi_c)/2 at
   *   the centre of a filament where phi reaches phi_c. Its sign gives the
   *   pinched phase (negative: phi > 0 is pinched).
   * - kappa*eps = -eps div(n) detects a fold of a single interface with a
   *   radius below epsilon. It is only used where the normal is well defined,
   *   |grad(phi)| > q/2, which excludes the midlines of filaments.
   *
   * Both are restricted to the reliable core |phi| < 0.95 of the interface.
   */
  template <int dim>
  InteractionIndicator
  evaluate_interaction_indicator(const double          phi,
                                 const Tensor<1, dim> &gradient,
                                 const Tensor<2, dim> &psi_hessian,
                                 const double          epsilon,
                                 const double          kappa_start = 1.)
  {
    constexpr double core_outer = interaction_core_outer, core_inner = 0.85;

    InteractionIndicator result;
    result.core =
      cutoff_ramp((core_outer - std::abs(phi)) / (core_outer - core_inner));
    if (result.core == 0.)
      return result;

    const double phi_c = std::clamp(phi, -core_outer, core_outer);
    const double w     = 1. - phi_c * phi_c;
    const double q     = w / (std::sqrt(2.) * epsilon);
    const double g     = gradient.norm();

    double psi_nn;
    if (g < 0.1 * q)
    {
      // Centre of a filament: the normal is undefined there, and the direction
      // of strongest curvature of psi plays its role.
      const auto ev = eigenvalues(symmetrize(psi_hessian));
      psi_nn        = *std::max_element(ev.begin(),
                                 ev.end(),
                                 [](const double a, const double b) {
                                   return std::abs(a) < std::abs(b);
                                 });
    }
    else
    {
      const Tensor<1, dim> n = gradient / g;
      psi_nn                 = n * (psi_hessian * n);
    }
    result.lambda        = 2. * epsilon * epsilon * psi_nn;
    result.kappa_defined = g > 0.5 * q;
    // The tangential part of the Hessian of psi is that of phi divided by w.
    if (result.kappa_defined)
      result.kappa_epsilon = -epsilon * w * (trace(psi_hessian) - psi_nn) / g;
    set_interaction_weights(result, kappa_start);
    return result;
  }

  /**
   * Continuous linear cutoff field chi, recomputed from phi at the start of
   * each time step and frozen during the Newton iterations:
   *
   * 0. continuous recovered gradient of psi = atanh(phi), whose derivative
   *    replaces the Hessian of phi in the indicators;
   * 1. raw weights per cell from the cell averages of Lambda and kappa*eps,
   *    separately for the two phases;
   * 2. dilation over about epsilon through vertex neighbours, so that a signal
   *    one cell wide is not diluted by the smoothing;
   * 3. Helmholtz smoothing chi - (eps/2)^2 lap(chi) = source, times two,
   *    clamped to [0,1];
   * 4. side mask: a pinched phase only cuts the correction on its own side of
   *    phi = 0, so that the outer flanks of a filament keep the correction;
   * 5. optional relaxation in time towards this detected value, with distinct
   *    rise and decay times. The value of the last accepted step is kept over
   *    rejected attempts and transferred when the mesh is adapted.
   *
   * The normal strain n.grad(u).n of the same state is averaged per cell over
   * the interface band, with n from the recovered gradient of psi. Its
   * positive part (stretching) gives the ratio r per cell, spread like chi by
   * the dilation and the Helmholtz smoothing.
   *
   * The values and gradients of chi and r at the assembly quadrature points
   * are tabulated per active cell. The table is discarded whenever the mesh
   * changes.
   */
  template <int dim>
  class ProfileCorrectionCutoff
  {
  public:
    struct Statistics
    {
      double                   max_chi     = 0.;
      types::global_cell_index n_cut_cells = 0;
      double                   max_strain  = 0.;
      double                   max_ratio   = 1.;
    };

    ProfileCorrectionCutoff(const Triangulation<dim>          &tria,
                            const InteractionCutoffParameters &params)
      : parameters(params)
      , dofs(tria)
    {
      connections.push_back(
        tria.signals.any_change.connect([this]() { n_q_points = 0; }));
      // With relaxation, chi carries a state from one step to the next.
      connections.push_back(tria.signals.pre_distributed_refinement.connect(
        [this]() { prepare_transfer(); }));
      connections.push_back(tria.signals.post_distributed_refinement.connect(
        [this]() { finish_transfer(); }));
    }

    ~ProfileCorrectionCutoff()
    {
      for (auto &connection : connections)
        connection.disconnect();
    }

    /**
     * True if the cutoff was computed on the current mesh for a quadrature of
     * n_q points.
     */
    const Triangulation<dim> &get_triangulation() const
    {
      return dofs.get_triangulation();
    }

    bool is_ready(const unsigned int n_q) const
    {
      return n_q_points > 0 && n_q == n_q_points;
    }

    double chi(const unsigned int cell_index, const unsigned int q) const
    {
      return chi_q[cell_index * n_q_points + q];
    }

    const Tensor<1, dim> &chi_gradient(const unsigned int cell_index,
                                       const unsigned int q) const
    {
      return chi_gradient_q[cell_index * n_q_points + q];
    }

    /**
     * Strain held by the unscaled correction, which follows its amplitude.
     */
    void set_strain_reference(const double strain)
    {
      parameters.strain_reference = strain;
    }

    /**
     * Weight r (1 - chi) of the correction coefficient and its gradient.
     */
    double weight(const unsigned int cell_index, const unsigned int q) const
    {
      const unsigned int i = cell_index * n_q_points + q;
      return ratio_q[i] * (1. - chi_q[i]);
    }

    Tensor<1, dim> weight_gradient(const unsigned int cell_index,
                                   const unsigned int q) const
    {
      const unsigned int i = cell_index * n_q_points + q;
      return (1. - chi_q[i]) * ratio_gradient_q[i] -
             ratio_q[i] * chi_gradient_q[i];
    }

    /**
     * Recompute chi from the phase field of the last accepted step. step and
     * dt identify the time step being attempted: several calls with the same
     * step (rejected attempts) relax from the same accepted value. The normal
     * strain and the ratio r need the velocity, of first component
     * velocity_component in the same solution.
     */
    template <typename VectorType>
    Statistics update(
      const DoFHandler<dim> &source,
      const Mapping<dim>    &mapping,
      const Quadrature<dim> &quadrature,
      const VectorType      &solution,
      const unsigned int     phase_component,
      const double           epsilon,
      const unsigned int     step,
      const double           dt,
      const unsigned int     velocity_component = numbers::invalid_unsigned_int)
    {
      const auto &tria    = dofs.get_triangulation();
      const auto  comm    = source.get_mpi_communicator();
      const bool  simplex = source.get_fe().reference_cell().is_simplex();
      if (!fe)
      {
        if (simplex)
          fe = std::make_unique<FE_SimplexP<dim>>(1);
        else
          fe = std::make_unique<FE_Q<dim>>(1);
      }
      // The numbering is deterministic: on an unchanged or just transferred
      // mesh, the previous chi keeps its meaning.
      dofs.distribute_dofs(*fe);
      const IndexSet owned    = dofs.locally_owned_dofs();
      const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dofs, constraints);
      constraints.close();

      // Value of chi at the last accepted step, the start of the relaxation.
      if (step != current_step)
      {
        accepted.reinit(owned, comm);
        if (field.size() == dofs.n_dofs() &&
            field.locally_owned_elements() == owned)
          accepted = field;
        current_step = step;
      }

      const FEValuesExtractors::Scalar     phase(phase_component);
      FEValues<dim>                        values(mapping,
                           source.get_fe(),
                           quadrature,
                           update_values | update_gradients |
                             update_JxW_values);
      FEValues<dim>                        linear(mapping,
                           *fe,
                           quadrature,
                           update_values | update_gradients |
                             update_JxW_values);
      const unsigned int                   n_q    = quadrature.size();
      const unsigned int                   n_dofs = fe->n_dofs_per_cell();
      std::vector<double>                  phi(n_q);
      std::vector<Tensor<1, dim>>          gradients(n_q);
      std::vector<types::global_dof_index> indices(n_dofs);

      // 0. Recovered gradient of psi = atanh(phi): lumped L2 projection on
      // the linear field. In a filament, the correction keeps each side on the
      // tanh of the distance to its own interface, on which the Hessian of psi
      // vanishes: the overlap is then only a kink of grad(phi) on the midline.
      // When the midline lies on a face (a filament aligned with the mesh, as
      // on a symmetry axis), the Hessian inside the elements never sees it,
      // while the derivative of the recovered gradient spreads it over a cell.
      std::array<LA::ParVectorType, dim> recovered;
      {
        LA::ParVectorType                  mass(owned, comm);
        std::array<LA::ParVectorType, dim> projected;
        Vector<double>                     local_mass(n_dofs);
        std::array<Vector<double>, dim>    local_gradient;
        for (unsigned int d = 0; d < dim; ++d)
        {
          projected[d].reinit(owned, comm);
          local_gradient[d].reinit(n_dofs);
        }
        for (const auto &cell : dofs.active_cell_iterators())
          if (cell->is_locally_owned())
          {
            values.reinit(cell->as_dof_handler_iterator(source));
            values[phase].get_function_values(solution, phi);
            values[phase].get_function_gradients(solution, gradients);
            linear.reinit(cell);
            local_mass = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              local_gradient[d] = 0.;
            for (unsigned int q = 0; q < n_q; ++q)
            {
              const auto psi = psi_gradient<dim>(phi[q], gradients[q]);
              for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const double m = linear.shape_value(i, q) * linear.JxW(q);
                local_mass(i) += m;
                for (unsigned int d = 0; d < dim; ++d)
                  local_gradient[d](i) += psi[d] * m;
              }
            }
            cell->get_dof_indices(indices);
            constraints.distribute_local_to_global(local_mass, indices, mass);
            for (unsigned int d = 0; d < dim; ++d)
              constraints.distribute_local_to_global(local_gradient[d],
                                                     indices,
                                                     projected[d]);
          }
        mass.compress(VectorOperation::add);
        for (unsigned int d = 0; d < dim; ++d)
        {
          projected[d].compress(VectorOperation::add);
          LA::ParVectorType nodal(owned, comm);
          // Constrained (hanging) dofs received nothing and are interpolated.
          for (const auto i : owned)
          {
            const double m = mass[i];
            if (m > 0.)
              nodal[i] = static_cast<double>(projected[d][i]) / m;
          }
          nodal.compress(VectorOperation::insert);
          constraints.distribute(nodal);
          recovered[d].reinit(owned, relevant, comm);
          recovered[d] = nodal;
        }
      }

      // 1. Raw weights per cell, and the excess r - 1 of the strain ratio as
      // a third component. The cell averages of the indicators and of the
      // normal strain are also kept for the visualization output.
      const bool with_strain =
        velocity_component != numbers::invalid_unsigned_int;
      const bool         scaling = with_strain && parameters.strain_scaling;
      const unsigned int n_cells = tria.n_active_cells();
      std::vector<std::array<double, 3>> weight(n_cells, {{0., 0., 0.}});
      lambda_cell.reinit(n_cells);
      kappa_cell.reinit(n_cells);
      strain_cell.reinit(with_strain ? n_cells : 0);
      std::array<std::vector<Tensor<1, dim>>, dim> recovered_gradients;
      std::array<std::vector<double>, dim>         recovered_values;
      for (unsigned int d = 0; d < dim; ++d)
      {
        recovered_gradients[d].resize(n_q);
        recovered_values[d].resize(n_q);
      }
      std::vector<Tensor<2, dim>> velocity_gradients(n_q);
      double                      h_min = std::numeric_limits<double>::max();
      Statistics                  stats;
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          values.reinit(cell->as_dof_handler_iterator(source));
          values[phase].get_function_values(solution, phi);
          values[phase].get_function_gradients(solution, gradients);
          linear.reinit(cell);
          h_min = std::min(h_min, cell->minimum_vertex_distance());
          if (with_strain)
          {
            // Normal strain averaged over the band, n from the recovered
            // gradient of psi, smoother than grad(phi) in the tails.
            values[FEValuesExtractors::Vector(velocity_component)]
              .get_function_gradients(solution, velocity_gradients);
            for (unsigned int d = 0; d < dim; ++d)
              linear.get_function_values(recovered[d], recovered_values[d]);
            double strain = 0., measure = 0.;
            for (unsigned int q = 0; q < n_q; ++q)
            {
              Tensor<1, dim> n;
              for (unsigned int d = 0; d < dim; ++d)
                n[d] = recovered_values[d][q];
              const double norm = n.norm();
              if (std::abs(phi[q]) >= parameters.strain_band || norm == 0.)
                continue;
              n /= norm;
              strain += n * (velocity_gradients[q] * n) * values.JxW(q);
              measure += values.JxW(q);
            }
            if (measure > 0.)
              strain /= measure;
            strain_cell[cell->active_cell_index()] = strain;
            stats.max_strain = std::max(stats.max_strain, strain);
            // Only the stretching spreads the isolines.
            if (scaling)
              weight[cell->active_cell_index()][2] =
                std::clamp(parameters.strain_safety * strain /
                             parameters.strain_reference,
                           1.,
                           parameters.max_strain_ratio) -
                1.;
          }
          if (!parameters.cutoff)
            continue;
          for (unsigned int d = 0; d < dim; ++d)
            linear.get_function_gradients(recovered[d], recovered_gradients[d]);
          // The pointwise values still carry an O(h/eps) error that
          // oscillates inside the elements, so the weights are built from the
          // cell averages of Lambda (over the core) and kappa*eps (where
          // defined).
          InteractionIndicator mean;
          double               core_measure = 0., kappa_measure = 0.;
          for (unsigned int q = 0; q < n_q; ++q)
          {
            Tensor<2, dim> hessian;
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int e = 0; e < dim; ++e)
                hessian[d][e] = 0.5 * (recovered_gradients[d][q][e] +
                                       recovered_gradients[e][q][d]);
            const auto indicator = evaluate_interaction_indicator<dim>(
              phi[q], gradients[q], hessian, epsilon, parameters.kappa_start);
            const double JxW = values.JxW(q);
            mean.core        = std::max(mean.core, indicator.core);
            mean.lambda += indicator.core * indicator.lambda * JxW;
            core_measure += indicator.core * JxW;
            if (indicator.kappa_defined)
            {
              mean.kappa_epsilon += indicator.kappa_epsilon * JxW;
              kappa_measure += JxW;
            }
          }
          if (core_measure > 0.)
            mean.lambda /= core_measure;
          mean.kappa_defined = kappa_measure > 0.;
          if (mean.kappa_defined)
            mean.kappa_epsilon /= kappa_measure;
          set_interaction_weights(mean, parameters.kappa_start);
          weight[cell->active_cell_index()][0]   = mean.positive;
          weight[cell->active_cell_index()][1]   = mean.negative;
          lambda_cell[cell->active_cell_index()] = mean.lambda;
          kappa_cell[cell->active_cell_index()]  = mean.kappa_epsilon;
        }
      h_min            = Utilities::MPI::min(h_min, comm);
      stats.max_strain = Utilities::MPI::max(stats.max_strain, comm);

      // 2. Dilation over about epsilon. Each layer takes the maximum over the
      // cells sharing a vertex, ghost values being refreshed before each layer.
      const unsigned int n_layers =
        std::clamp(static_cast<unsigned int>(std::ceil(epsilon / h_min)),
                   1u,
                   3u);
      const auto vertex_to_cells = GridTools::vertex_to_cell_map(tria);
      for (unsigned int layer = 0; layer < n_layers; ++layer)
      {
        GridTools::exchange_cell_data_to_ghosts<std::array<double, 3>>(
          dofs,
          [&](const typename DoFHandler<dim>::active_cell_iterator &cell)
            -> std::optional<std::array<double, 3>> {
            return weight[cell->active_cell_index()];
          },
          [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
              const std::array<double, 3>                          &data) {
            weight[cell->active_cell_index()] = data;
          });
        auto dilated = weight;
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
          {
            auto &d = dilated[cell->active_cell_index()];
            for (const unsigned int v : cell->vertex_indices())
              for (const auto &neighbor :
                   vertex_to_cells[cell->vertex_index(v)])
                for (unsigned int c = 0; c < 3; ++c)
                  d[c] =
                    std::max(d[c], weight[neighbor->active_cell_index()][c]);
          }
        weight = std::move(dilated);
      }

      // 3. Helmholtz smoothing with the dilated cell values as source.
      DynamicSparsityPattern dsp(relevant);
      DoFTools::make_sparsity_pattern(dofs, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp, owned, comm, relevant);
      LA::ParMatrixType matrix;
      matrix.reinit(owned, owned, dsp, comm);
      std::array<LA::ParVectorType, 3> rhs, smoothed;
      for (unsigned int c = 0; c < 3; ++c)
      {
        rhs[c].reinit(owned, comm);
        smoothed[c].reinit(owned, comm);
      }
      const double                  ell2 = 0.25 * epsilon * epsilon;
      FullMatrix<double>            local_matrix(n_dofs, n_dofs);
      std::array<Vector<double>, 3> local_rhs = {{Vector<double>(n_dofs),
                                                  Vector<double>(n_dofs),
                                                  Vector<double>(n_dofs)}};
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          linear.reinit(cell);
          local_matrix  = 0.;
          local_rhs[0]  = 0.;
          local_rhs[1]  = 0.;
          local_rhs[2]  = 0.;
          const auto &w = weight[cell->active_cell_index()];
          for (unsigned int q = 0; q < n_q; ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double phi_i = linear.shape_value(i, q);
              for (unsigned int j = 0; j < n_dofs; ++j)
                local_matrix(i, j) +=
                  (phi_i * linear.shape_value(j, q) +
                   ell2 * (linear.shape_grad(i, q) * linear.shape_grad(j, q))) *
                  linear.JxW(q);
              for (unsigned int c = 0; c < 3; ++c)
                local_rhs[c](i) += w[c] * phi_i * linear.JxW(q);
            }
          cell->get_dof_indices(indices);
          constraints.distribute_local_to_global(
            local_matrix, local_rhs[0], indices, matrix, rhs[0]);
          for (unsigned int c = 1; c < 3; ++c)
            constraints.distribute_local_to_global(local_rhs[c],
                                                   indices,
                                                   rhs[c]);
        }
      matrix.compress(VectorOperation::add);
      for (unsigned int c = 0; c < 3; ++c)
      {
        rhs[c].compress(VectorOperation::add);
        if (rhs[c].l2_norm() == 0.)
          continue;
        SolverControl control(1000, 1e-10 * rhs[c].l2_norm());
        LA::SolverCG  cg(control);
        PETScWrappers::PreconditionBlockJacobi preconditioner(matrix);
        cg.solve(matrix, smoothed[c], rhs[c], preconditioner);
        constraints.distribute(smoothed[c]);
      }

      // 4. Phase values at the vertices, then side mask and clamp.
      LA::ParVectorType   nodal_phi(owned, comm);
      FEValues<dim>       at_vertices(mapping,
                                source.get_fe(),
                                Quadrature<dim>(fe->get_unit_support_points()),
                                update_values);
      std::vector<double> vertex_phi(n_dofs);
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          at_vertices.reinit(cell->as_dof_handler_iterator(source));
          at_vertices[phase].get_function_values(solution, vertex_phi);
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < n_dofs; ++i)
            if (owned.is_element(indices[i]))
              nodal_phi[indices[i]] = vertex_phi[i];
        }
      nodal_phi.compress(VectorOperation::insert);
      LA::ParVectorType chi_owned(owned, comm);
      for (const auto i : owned)
      {
        const double p      = nodal_phi[i];
        const double pos    = std::clamp(2. * smoothed[0][i], 0., 1.);
        const double neg    = std::clamp(2. * smoothed[1][i], 0., 1.);
        const double target = std::max(pos * cutoff_ramp((p + 0.25) / 0.5),
                                       neg * cutoff_ramp((0.25 - p) / 0.5));
        // 5. First-order relaxation in time, faster to cut than to restore.
        const double previous = accepted[i];
        const double tau =
          target > previous ? parameters.rise_time : parameters.decay_time;
        chi_owned[i] =
          tau > 0. ? previous + dt / (tau + dt) * (target - previous) : target;
      }
      chi_owned.compress(VectorOperation::insert);
      constraints.distribute(chi_owned);
      field.reinit(owned, relevant, comm);
      field = chi_owned;

      // Strain ratio r = 1 + smoothed excess, at most the maximum ratio.
      LA::ParVectorType ratio_owned(owned, comm);
      for (const auto i : owned)
        ratio_owned[i] = 1. + std::clamp(static_cast<double>(smoothed[2][i]),
                                         0.,
                                         parameters.max_strain_ratio - 1.);
      ratio_owned.compress(VectorOperation::insert);
      constraints.distribute(ratio_owned);
      ratio.reinit(owned, relevant, comm);
      ratio = ratio_owned;

      // Tabulate chi, r and their gradients at the assembly quadrature points.
      chi_q.assign(n_cells * n_q, 0.);
      chi_gradient_q.assign(n_cells * n_q, Tensor<1, dim>());
      ratio_q.assign(n_cells * n_q, 1.);
      ratio_gradient_q.assign(n_cells * n_q, Tensor<1, dim>());
      std::vector<double>         chi_values(n_q), ratio_values(n_q);
      std::vector<Tensor<1, dim>> chi_gradients(n_q), ratio_gradients(n_q);
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          linear.reinit(cell);
          linear.get_function_values(field, chi_values);
          linear.get_function_gradients(field, chi_gradients);
          linear.get_function_values(ratio, ratio_values);
          linear.get_function_gradients(ratio, ratio_gradients);
          const unsigned int offset   = cell->active_cell_index() * n_q;
          double             cell_max = 0.;
          for (unsigned int q = 0; q < n_q; ++q)
          {
            chi_q[offset + q]            = chi_values[q];
            chi_gradient_q[offset + q]   = chi_gradients[q];
            ratio_q[offset + q]          = ratio_values[q];
            ratio_gradient_q[offset + q] = ratio_gradients[q];
            cell_max                     = std::max(cell_max, chi_values[q]);
            stats.max_ratio = std::max(stats.max_ratio, ratio_values[q]);
          }
          stats.max_chi = std::max(stats.max_chi, cell_max);
          if (cell_max > 0.5)
            ++stats.n_cut_cells;
        }
      n_q_points        = n_q;
      stats.max_chi     = Utilities::MPI::max(stats.max_chi, comm);
      stats.n_cut_cells = Utilities::MPI::sum(stats.n_cut_cells, comm);
      stats.max_ratio   = Utilities::MPI::max(stats.max_ratio, comm);
      return stats;
    }

    /**
     * Add chi, the cell averages of Lambda and kappa eps, the normal strain
     * and the ratio r to an output object exposing add_data_vector and
     * add_cell_data_vector.
     */
  private:
    bool relaxation() const
    {
      return parameters.rise_time > 0. || parameters.decay_time > 0.;
    }

    // Called at the start of the refinement: register chi for the transfer.
    void prepare_transfer()
    {
      if (!relaxation() || !fe || dofs.n_dofs() == 0 ||
          field.size() != dofs.n_dofs())
        return;
      transfer =
        std::make_unique<SolutionTransfer<dim, LA::ParVectorType>>(dofs);
      transfer->prepare_for_coarsening_and_refinement(field);
    }

    // Called once the mesh has reached its final state: recover chi on it.
    void finish_transfer()
    {
      if (!transfer)
        return;
      const auto comm = dofs.get_mpi_communicator();
      dofs.distribute_dofs(*fe);
      const IndexSet owned    = dofs.locally_owned_dofs();
      const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      LA::ParVectorType transferred(owned, comm);
      transfer->interpolate(transferred);
      AffineConstraints<double> constraints(owned, relevant);
      DoFTools::make_hanging_node_constraints(dofs, constraints);
      constraints.close();
      constraints.distribute(transferred);
      field.reinit(owned, relevant, comm);
      field = transferred;
      transfer.reset();
    }

  public:
    template <typename OutputType>
    void add_data(OutputType &output) const
    {
      if (n_q_points == 0)
        return;
      output.add_data_vector(
        dofs,
        field,
        std::vector<std::string>{"profile_correction_cutoff"},
        std::vector<DataComponentInterpretation::DataComponentInterpretation>{
          DataComponentInterpretation::component_is_scalar});
      if (parameters.cutoff)
      {
        output.add_cell_data_vector(lambda_cell, "interaction_lambda");
        output.add_cell_data_vector(kappa_cell, "interaction_kappa_epsilon");
      }
      if (strain_cell.size() > 0)
        output.add_cell_data_vector(strain_cell, "interface_normal_strain");
      if (parameters.strain_scaling)
        output.add_data_vector(
          dofs,
          ratio,
          std::vector<std::string>{"profile_correction_strain_ratio"},
          std::vector<DataComponentInterpretation::DataComponentInterpretation>{
            DataComponentInterpretation::component_is_scalar});
    }

  private:
    InteractionCutoffParameters         parameters;
    std::unique_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dofs;
    LA::ParVectorType                   field;
    // chi at the last accepted step, and the step it belongs to.
    LA::ParVectorType accepted;
    unsigned int      current_step = numbers::invalid_unsigned_int;
    std::unique_ptr<SolutionTransfer<dim, LA::ParVectorType>> transfer;
    Vector<float>                                             lambda_cell;
    Vector<float>                                             kappa_cell;
    // Normal strain per cell (empty without velocity) and ratio r.
    Vector<float>                            strain_cell;
    LA::ParVectorType                        ratio;
    std::vector<double>                      ratio_q;
    std::vector<Tensor<1, dim>>              ratio_gradient_q;
    std::vector<double>                      chi_q;
    std::vector<Tensor<1, dim>>              chi_gradient_q;
    unsigned int                             n_q_points = 0;
    std::vector<boost::signals2::connection> connections;
  };
} // namespace CahnHilliard

#endif
