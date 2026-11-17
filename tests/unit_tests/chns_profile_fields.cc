#include <chns_profile_correction_output.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

#include "../tests.h"
using namespace dealii;
int main()
{
  initlog();
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FESystem<2>   fe(FE_Q<2>(1), 5);
  DoFHandler<2> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<2>          mapping;
  std::vector<Point<2>> points(dofs.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dofs, points);
  Vector<double>                       solution(dofs.n_dofs());
  const auto                           cell = dofs.begin_active();
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  cell->get_dof_indices(indices);
  for (unsigned j = 0; j < indices.size(); ++j)
  {
    const auto i = indices[j];
    const auto c = fe.system_to_component_index(j).first;
    solution[i]  = c == 0 ? 2. * points[i][0] :
                   c == 1 ? -points[i][1] :
                   c == 3 ? std::tanh(.2 + .3 * points[i][0]) :
                            0.;
  }
  Parameters::CahnHilliard<2> ch;
  ch.enable_profile_correction           = true;
  ch.mobility                            = .2;
  ch.surface_tension                     = .8;
  ch.epsilon_interface                   = .1;
  ch.profile_correction_timescale_factor = 1.3;
  DataPostprocessorInputs::Vector<2> input;
  input.set_cell<2>(cell);
  input.evaluation_points = {Point<2>(.3, .4)};

  for (bool local : {false, true})
  {
    ch.profile_correction_model =
      local ?
        Parameters::CahnHilliard<2>::ProfileCorrectionModel::local_timescale :
        Parameters::CahnHilliard<2>::ProfileCorrectionModel::constant;
    std::vector<Vector<double>> out(1, Vector<double>(local ? 11 : 4));
    const double base = .2 * (3. / (2. * std::sqrt(2.))) * .8 / .1;
    CahnHilliard::ProfileCorrectionOutput<2, Vector<double>> output(
      ch,
      mapping,
      solution,
      3,
      0,
      {},
      2. * ch.profile_correction_strength * base);
    output.evaluate_vector_field(input, out);
    const double z     = 2. * .3 * .3 / (.3 * .3 + .05 * .05 / (2. * .1 * .1));
    const double delta = 1e-8 * base / (.1 * .1);
    const double K     = local ?
                           1.3 * (base + .1 * .1 * (std::hypot(z, delta) - delta)) :
                           2. * ch.profile_correction_strength * base;
    const double r     = std::tanh(.2 + .3 * .3);
    Tensor<1, 2> grad;
    grad[0]      = (1. - r * r) * .3;
    const auto f = CahnHilliard::evaluate_profile_correction(
      r, grad, Tensor<2, 2>(), .1, false);
    AssertThrow(std::abs(out[0][0] - K / (2. * base)) < 1e-12,
                ExcMessage("Exported strength"));
    AssertThrow(std::abs(out[0][1] - K) < 1e-12,
                ExcMessage("Exported coefficient"));
    for (unsigned d = 0; d < 2; ++d)
      AssertThrow(std::abs(out[0][2 + d] - K * f.flux[d]) < 1e-12,
                  ExcMessage("Exported reconstructed flux"));
    AssertThrow(output.get_names().size() == (local ? 11 : 4),
                ExcInternalError());
    DataOut<2> volume;
    volume.attach_dof_handler(dofs);
    volume.add_data_vector(solution, output);
    volume.build_patches(mapping, 2);
    std::ostringstream vtu;
    volume.write_vtu(vtu);
    AssertThrow(vtu.str().find("profile_correction_strength") !=
                    std::string::npos &&
                  vtu.str().find("profile_correction_flux") !=
                    std::string::npos,
                ExcMessage("Missing volume fields"));
    DataOutFaces<2> skin;
    skin.attach_dof_handler(dofs);
    skin.add_data_vector(solution, output);
    skin.build_patches(mapping, 2);
    std::ostringstream skin_vtu;
    skin.write_vtu(skin_vtu);
    AssertThrow(skin_vtu.str().find("profile_correction_coefficient") !=
                  std::string::npos,
                ExcMessage("Missing skin fields"));
  }
  deallog << "Constant and local profile output matches reconstructed flux OK"
          << std::endl;
}
