#ifndef CHNS_PROFILE_GEOMETRY_BUILDER_H
#define CHNS_PROFILE_GEOMETRY_BUILDER_H
#include <chns_profile_geometry.h>
#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <parameter_reader.h>

namespace CHNSProfileOverlap
{
  template <int dim>
  class Contour
  {
  public:
    template <typename VectorType>
    void compute(const dealii::DoFHandler<dim> &dofs,
                 const dealii::Mapping<dim>    &mapping,
                 const VectorType              &solution,
                 unsigned                       phase_component,
                 double                         epsilon,
                 double                         radius_in_epsilon,
                 unsigned                       subdivisions,
                 double                         contour_level = 0.)
    {
      using namespace dealii;
      AssertThrow(dim == 2,
                  ExcMessage(
                    "Interface proximity is currently implemented in 2D only"));
      AssertThrow(subdivisions > 0 &&
                    phase_component < dofs.get_fe().n_components(),
                  ExcMessage("Invalid interface proximity sampling/component"));
      eps         = epsilon;
      phase_index = phase_component;
      if constexpr (dim == 2)
      {
        const bool simplex = dofs.get_fe().reference_cell().is_simplex();
        std::vector<dealii::Point<dim>>      points;
        std::vector<std::array<unsigned, 3>> triangles;
        const unsigned                       n = subdivisions;
        std::vector<std::vector<unsigned>>   ids(n + 1,
                                               std::vector<unsigned>(n + 1));
        for (unsigned i = 0; i <= n; ++i)
          for (unsigned j = 0; j <= n; ++j)
            if (!simplex || i + j <= n)
            {
              ids[i][j] = points.size();
              points.emplace_back(double(i) / n, double(j) / n);
            }
        for (unsigned i = 0; i < n; ++i)
          for (unsigned j = 0; j < n; ++j)
            if (!simplex || i + j < n)
            {
              triangles.push_back({{ids[i][j], ids[i + 1][j], ids[i][j + 1]}});
              if (!simplex || i + j + 1 < n)
                triangles.push_back(
                  {{ids[i + 1][j], ids[i + 1][j + 1], ids[i][j + 1]}});
            }
        FEValues<dim>                    fe(mapping,
                         dofs.get_fe(),
                         Quadrature<dim>(points),
                         update_values | update_quadrature_points);
        std::vector<double>              values(points.size()), packed;
        Vector<double>                   nodal(dofs.get_fe().n_dofs_per_cell());
        const FEValuesExtractors::Scalar phase(phase_component);
        for (const auto &cell : dofs.active_cell_iterators())
          if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            if (contour_level == 0.)
              fe[phase].get_function_values(solution, values);
            else
            {
              // Partition of unity: interpolate (phi_i - level), rather than
              // subtracting level from an interpolated value close to +/-1.
              // This avoids cancellation of O(1) numbers in the flat tails.
              // Only the contour arithmetic changes, never phi or the flux.
              cell->get_dof_values(solution, nodal);
              std::fill(values.begin(), values.end(), 0.);
              for (unsigned j = 0; j < nodal.size(); ++j)
                if (dofs.get_fe().system_to_component_index(j).first ==
                    phase_component)
                  for (unsigned q = 0; q < values.size(); ++q)
                    values[q] +=
                      (nodal[j] - contour_level) * fe[phase].value(j, q);
            }
            const auto &physical = fe.get_quadrature_points();
            for (const auto &tri : triangles)
            {
              std::array<Point, 3>  p;
              std::array<double, 3> v;
              for (unsigned j = 0; j < 3; ++j)
              {
                p[j] = {physical[tri[j]][0], physical[tri[j]][1]};
                v[j] = values[tri[j]];
              }
              if (v[0] == 0. && v[1] == 0. && v[2] == 0.)
                continue;
              std::vector<Point> roots;
              const auto         append = [&](Point q) {
                for (auto r : roots)
                  if (dot(subtract(q, r), subtract(q, r)) < 1e-24 * eps * eps)
                    return;
                roots.push_back(q);
              };
              for (unsigned j = 0; j < 3; ++j)
              {
                const unsigned k = (j + 1) % 3;
                if (v[j] == 0.)
                  append(p[j]);
                if ((v[j] < 0. && v[k] > 0.) || (v[j] > 0. && v[k] < 0.))
                  append(
                    add(p[j],
                        scale(subtract(p[k], p[j]), -v[j] / (v[k] - v[j]))));
              }
              if (roots.size() != 2)
                continue;
              const Point  a = subtract(p[1], p[0]), b = subtract(p[2], p[0]);
              const double det = cross(a, b);
              if (det == 0.)
                continue;
              const Point gradient{
                ((v[1] - v[0]) * b[1] - (v[2] - v[0]) * a[1]) / det,
                (a[0] * (v[2] - v[0]) - b[0] * (v[1] - v[0])) / det};
              Point edge = subtract(roots[1], roots[0]);
              Point normal{-edge[1], edge[0]};
              if (dot(normal, gradient) < 0.)
                normal = scale(normal, -1.);
              if (dot(normal, normal) <= 1e-24 * eps * eps)
                continue;
              packed.insert(packed.end(),
                            {roots[0][0],
                             roots[0][1],
                             roots[1][0],
                             roots[1][1],
                             normal[0],
                             normal[1]});
            }
          }
        const auto gathered =
          Utilities::MPI::all_gather(dofs.get_mpi_communicator(), packed);
        std::vector<Segment> segments;
        for (const auto &part : gathered)
          for (std::size_t i = 0; i < part.size(); i += 6)
            segments.push_back({{part[i], part[i + 1]},
                                {part[i + 2], part[i + 3]},
                                {part[i + 4], part[i + 5]}});
        geometry =
          std::make_unique<Geometry>(std::move(segments),
                                     eps,
                                     radius_in_epsilon,
                                     contour_level == 0. ?
                                       0. :
                                       -std::copysign(1., contour_level));
      }
    }

    std::vector<MidlineSource>
    midline_sources(double level,
                    bool   weight_overlap  = false,
                    bool   confine_to_film = false) const
    {
      return geometry ? geometry->midline_sources(level,
                                                  weight_overlap,
                                                  confine_to_film) :
                        std::vector<MidlineSource>{};
    }

  private:
    double                    eps         = 1.;
    unsigned                  phase_index = 0;
    std::unique_ptr<Geometry> geometry;
  };
} // namespace CHNSProfileOverlap
namespace CHNSProfileWidth
{
  template <int dim, typename VectorType>
  std::shared_ptr<const Geometry>
  build(const dealii::DoFHandler<dim>           &dofs,
        const dealii::Mapping<dim>              &mapping,
        const VectorType                        &solution,
        unsigned                                 phase_component,
        double                                   eps,
        double                                   C,
        const CHNSProfileOverlap::SmoothPenalty &penalty_field)
  {
    using namespace dealii;
    if constexpr (dim != 2)
      throw std::invalid_argument(
        "Local timescale correction requires fixed two-dimensional CHNS");
    else
    {
      AssertThrow(dofs.get_fe().reference_cell().is_hyper_cube(),
                  ExcMessage(
                    "Local timescale correction requires quadrilateral cells"));
      const auto                          comm = dofs.get_mpi_communicator();
      const std::vector<dealii::Point<2>> points{{0., 0.},
                                                 {1., 0.},
                                                 {0., 1.},
                                                 {1., 1.}};
      FEValues<2>                         fe(mapping,
                     dofs.get_fe(),
                     Quadrature<2>(points),
                     update_values | update_quadrature_points);
      const FEValuesExtractors::Scalar    phase(phase_component);
      std::vector<double>                 values(4), packed;
      unsigned                            invalid = 0;
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[phase].get_function_values(solution, values);
          const auto  &p   = fe.get_quadrature_points();
          const double tol = 1e-10 * eps;
          if (std::abs(p[0][1] - p[1][1]) > tol ||
              std::abs(p[0][0] - p[2][0]) > tol ||
              std::abs(p[3][0] - p[1][0]) > tol ||
              std::abs(p[3][1] - p[2][1]) > tol || p[3][0] <= p[0][0] ||
              p[3][1] <= p[0][1])
            invalid = 1;
          packed.insert(packed.end(),
                        {p[0][0],
                         p[0][1],
                         p[3][0],
                         p[3][1],
                         values[0],
                         values[1],
                         values[2],
                         values[3]});
        }
      AssertThrow(
        Utilities::MPI::max(invalid, comm) == 0,
        ExcMessage(
          "Local timescale correction width feedback currently requires "
          "axis-aligned rectangular cells"));
      const auto        global = Utilities::MPI::all_gather(comm, packed);
      std::vector<Cell> cells;
      for (const auto &part : global)
        for (std::size_t i = 0; i < part.size(); i += 8)
          cells.push_back(
            {{part[i], part[i + 1]},
             {part[i + 2], part[i + 3]},
             {{part[i + 4], part[i + 5], part[i + 6], part[i + 7]}}});
      auto       geometry = std::make_shared<Geometry>(std::move(cells), eps);
      const auto penalty  = [&](Point p) {
        return penalty_field.evaluate(p).value;
      };
      const auto local =
        geometry->measure(Utilities::MPI::this_mpi_process(comm),
                          Utilities::MPI::n_mpi_processes(comm),
                          penalty);
      geometry->install(Utilities::MPI::all_gather(comm, local), C);
      return geometry;
    }
  }
} // namespace CHNSProfileWidth
namespace CHNSProfileGeometry
{
  /** Rebuild the validated overlap/width fields from the supplied phase
   * snapshot. */
  template <int dim, typename VectorType>
  std::shared_ptr<const Fields> build(const dealii::DoFHandler<dim> &dofs,
                                      const dealii::Mapping<dim>    &mapping,
                                      const VectorType              &solution,
                                      unsigned phase_component,
                                      const ParameterReader<dim> &param)
  {
    if constexpr (dim != 2)
      throw std::invalid_argument(
        "Local timescale correction requires 2D CHNS");
    else
    {
      using namespace dealii;
      const auto comm   = dofs.get_mpi_communicator();
      auto       fields = std::make_shared<Fields>();
      using namespace CHNSProfileOverlap;
      std::vector<MidlineSource> sources;
      // Keep the same zero-equivalent interaction range, but locate partners on
      // better-conditioned contours. The physical correction tail is unchanged.
      const double detector_level = .99;
      if (param.cahn_hilliard.enable_profile_correction)
        for (const double level : {-detector_level, detector_level})
        {
          Contour<dim> contour;
          contour.compute(dofs,
                          mapping,
                          solution,
                          phase_component,
                          param.cahn_hilliard.epsilon_interface,
                          32.,
                          4,
                          level);
          const auto next = contour.midline_sources(level, true, true);
          sources.insert(sources.end(), next.begin(), next.end());
        }
      ReflectingBox box;
      {
        // Even extension of the kernel at straight, natural phase boundaries.
        // This corrects kernel truncation, without changing any physical BC.
        std::array<double, 2> lo{{1e100, 1e100}}, hi{{-1e100, -1e100}};
        for (const auto &cell : dofs.active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto v : cell->vertex_indices())
              for (unsigned d = 0; d < 2; ++d)
              {
                lo[d] = std::min(lo[d], cell->vertex(v)[d]);
                hi[d] = std::max(hi[d], cell->vertex(v)[d]);
              }
        for (unsigned d = 0; d < 2; ++d)
        {
          lo[d] = Utilities::MPI::min(lo[d], comm);
          hi[d] = Utilities::MPI::max(hi[d], comm);
        }
        box.lower = {lo[0], lo[1]};
        box.upper = {hi[0], hi[1]};
        std::array<unsigned, 4> found{}, blocked{};
        const double tol = 1e-8 * param.cahn_hilliard.epsilon_interface;
        for (const auto &cell : dofs.active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary())
              {
                const auto f  = cell->face(face);
                const auto it = param.cahn_hilliard_bc.find(f->boundary_id());
                const bool natural =
                  it == param.cahn_hilliard_bc.end() ||
                  it->second.type == BoundaryConditions::Type::no_flux ||
                  it->second.type == BoundaryConditions::Type::none;
                for (unsigned d = 0; d < 2; ++d)
                  for (unsigned side = 0; side < 2; ++side)
                  {
                    bool         on_plane = true;
                    const double plane    = side ? hi[d] : lo[d];
                    for (const auto v : f->vertex_indices())
                      on_plane =
                        on_plane && std::abs(f->vertex(v)[d] - plane) < tol;
                    if (on_plane)
                    {
                      ++found[2 * d + side];
                      if (!natural)
                        ++blocked[2 * d + side];
                    }
                  }
              }
        for (unsigned d = 0; d < 2; ++d)
          for (unsigned side = 0; side < 2; ++side)
          {
            const bool reflect =
              Utilities::MPI::sum(found[2 * d + side], comm) > 0 &&
              Utilities::MPI::sum(blocked[2 * d + side], comm) == 0;
            if (side)
              box.reflect_upper[d] = reflect;
            else
              box.reflect_lower[d] = reflect;
          }
      }
      fields->overlap = std::make_shared<const SmoothPenalty>(
        std::move(sources),
        param.cahn_hilliard.epsilon_interface,
        std::sqrt(2.),
        box);
      fields->width = CHNSProfileWidth::build(
        dofs,
        mapping,
        solution,
        phase_component,
        param.cahn_hilliard.epsilon_interface,
        param.cahn_hilliard.profile_correction_timescale_factor,
        *fields->overlap);
      return fields;
    }
  }
} // namespace CHNSProfileGeometry
#endif
