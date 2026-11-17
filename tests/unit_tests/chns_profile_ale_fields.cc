#include <chns_profile_correction_output.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

#include "../tests.h"

using namespace dealii;

void check_mesh_coefficient()
{
  using Coefficient = CahnHilliard::LocalProfileCoefficient<2>;
  CHNSProfileOverlap::PenaltyValue geometry;
  geometry.value               = 3.;
  geometry.gradient            = {.4, -.3};
  geometry.feedback.multiplier = 2.3;
  geometry.feedback.gradient   = {-.2, .6};
  Tensor<2, 2> G, U;
  G[0][0]            = .2;
  G[0][1]            = -.1;
  G[1][0]            = .3;
  G[1][1]            = .4;
  U[0][0]            = .3;
  U[1][1]            = -.5;
  constexpr double h = 1e-6, eps = .2;
  const auto       transported = [&](double step) {
    auto result = geometry;
    auto J      = step * G;
    J[0][0] += 1.;
    J[1][1] += 1.;
    const auto transform = transpose(invert(J));
    const auto covector  = [&](CHNSProfileOverlap::Point p) {
      Tensor<1, 2> v;
      v[0] = p.x;
      v[1] = p.y;
      v    = transform * v;
      return CHNSProfileOverlap::Point{v[0], v[1]};
    };
    result.gradient          = covector(geometry.gradient);
    result.feedback.gradient = covector(geometry.feedback.gradient);
    return result;
  };
  for (const double ratio : {.8, 1.2})
  {
    Tensor<1, 2> w;
    w[0] = ratio / (std::sqrt(2.) * eps);
    const Coefficient a(w, {}, U, {}, eps, .7, 50., geometry);
    const auto        da = a.variation({}, {}, {}, {}, G);
    const Coefficient plus(w, {}, U, {}, eps, .7, 50., transported(h)),
      minus(w, {}, U, {}, eps, .7, 50., transported(-h));
    AssertThrow(
      std::abs((plus.value - minus.value) / (2. * h) - da.value) < 2e-7 &&
        ((plus.gradient - minus.gradient) / (2. * h) - da.gradient).norm() <
          2e-7,
      ExcMessage("Mesh derivative of nonzero overlap/width covectors"));
    AssertThrow(da.gradient.norm() > .1,
                ExcMessage("Inactive mesh covector regression"));
  }
}

int main()
{
  initlog();
  check_mesh_coefficient();
  Triangulation<2> quad, tria;
  GridGenerator::hyper_cube(quad);
  GridGenerator::convert_hypercube_to_simplex_mesh(quad, tria, 2);
  FESystem<2>   fe(FE_SimplexP<2>(1), 7);
  DoFHandler<2> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingFE<2>          fixed(FE_SimplexP<2>(1));
  std::vector<Point<2>> points(dofs.n_dofs());
  DoFTools::map_dofs_to_support_points(fixed, dofs, points);
  Vector<double> solution(dofs.n_dofs());
  auto           geometry = std::make_shared<CHNSProfileGeometry::Fields>();
  geometry->overlap       = std::make_shared<CHNSProfileOverlap::SmoothPenalty>(
    std::vector<CHNSProfileOverlap::MidlineSource>{
      {{.1, -.5}, {.1, 1.5}, .003}},
    .1,
    std::sqrt(2.));
  for (const auto &cell : dofs.active_cell_iterators())
  {
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    cell->get_dof_indices(indices);
    for (unsigned j = 0; j < indices.size(); ++j)
    {
      const auto i = indices[j], c = fe.system_to_component_index(j).first;
      const auto p = points[i];
      solution[i]  = c == 0 ? 2. * p[0] :
                     c == 1 ? -p[1] :
                     c == 3 ? 2. * p[0] + 3. :
                     c == 4 ? 1.5 * p[1] + 4. :
                     c == 5 ? std::tanh(.2 + .3 * p[0]) :
                              0.;
    }
    CHNSProfileGeometry::CellMap map;
    map.simplex = true;
    for (unsigned i = 0; i < 3; ++i)
      map.nodes[i] = {cell->vertex(i)[0], cell->vertex(i)[1]};
    geometry->reference_cells.emplace(cell->id().to_string(), map);
  }
  MappingFEField<2, 2, Vector<double>> moving(
    dofs, solution, fe.component_mask(FEValuesExtractors::Vector(3)));
  Parameters::CahnHilliard<2> ch;
  ch.enable_profile_correction = true;
  ch.profile_correction_model =
    Parameters::CahnHilliard<2>::ProfileCorrectionModel::local_timescale;
  ch.mobility                            = .2;
  ch.surface_tension                     = .8;
  ch.epsilon_interface                   = .1;
  ch.profile_correction_timescale_factor = 1.3;
  CahnHilliard::ProfileCorrectionOutput<2, Vector<double>> output(
    ch, moving, solution, 5, 0, geometry, 0.);
  for (const auto &cell : dofs.active_cell_iterators())
  {
    const Point<2> reference(.23, .19);
    const auto     old = fixed.transform_unit_to_real_cell(cell, reference);
    DataPostprocessorInputs::Vector<2> input;
    input.set_cell<2>(cell);
    input.evaluation_points = {
      moving.transform_unit_to_real_cell(cell, reference)};
    std::vector<Vector<double>> out(1, Vector<double>(11));
    output.evaluate_vector_field(input, out);
    const double penalty = geometry->evaluate(old).value;
    const double base    = .2 * (3. / (2. * std::sqrt(2.))) * .8 / .1;
    const double z       = .15 * .15 / (.15 * .15 + .05 * .05 / (2. * .1 * .1));
    const double delta   = 1e-8 * base / (.1 * .1);
    const double K =
      1.3 * (base + .1 * .1 * (std::hypot(z, delta) - delta)) / (1. + penalty);
    AssertThrow(std::abs(out[0][1] - K) < 1e-12 &&
                  std::abs(out[0][4] - penalty) < 1e-12,
                ExcMessage("ALE export sampled geometry in the current instead "
                           "of the frozen configuration"));
    const double phi = std::tanh(.2 + .3 * old[0]);
    Tensor<1, 2> g;
    g[0]            = (1. - phi * phi) * .15;
    const auto flux = CahnHilliard::evaluate_profile_correction(
      phi, g, Tensor<2, 2>(), .1, false);
    AssertThrow(std::abs(out[0][2] - K * flux.flux[0]) < 1e-12 &&
                  std::abs(out[0][3]) < 1e-12,
                ExcMessage("ALE export flux reconstruction"));
  }
  DataOut<2> volume;
  volume.attach_dof_handler(dofs);
  volume.add_data_vector(solution, output);
  volume.build_patches(moving, 2);
  DataOutFaces<2> skin;
  skin.attach_dof_handler(dofs);
  skin.add_data_vector(solution, output);
  skin.build_patches(moving, 2);
  deallog << "ALE triangle volume/skin exports use the frozen geometry and "
             "current gradients OK"
          << std::endl;
}
