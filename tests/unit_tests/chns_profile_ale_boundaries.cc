#include <chns_profile_geometry_builder.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>

#include "../tests.h"

using namespace dealii;

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            std::vector<unsigned>{2, 1},
                                            Point<2>(),
                                            Point<2>(1., 1.));
  FESystem<2>   fe(FE_Q<2>(2), 2, FE_Q<2>(1), 1);
  DoFHandler<2> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<2>          fixed;
  std::vector<Point<2>> points(dofs.n_dofs());
  DoFTools::map_dofs_to_support_points(fixed, dofs, points);
  Vector<double> solution(dofs.n_dofs());
  for (const auto &cell : dofs.active_cell_iterators())
  {
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    cell->get_dof_indices(indices);
    for (unsigned j = 0; j < indices.size(); ++j)
    {
      const auto i = indices[j], c = fe.system_to_component_index(j).first;
      const auto p = points[i];
      // Each valid Q2 map has determinant >= .04 times the original one.
      // The outer boundary is exactly the unit square, but a Bernstein
      // control point is at y=-.46. It must not define the reflection plane.
      const double x = p[0] - .5 * std::floor(2. * p[0]);
      solution[i]    = c == 0 ? p[0] :
                       c == 1 ? p[1] - 15.36 * x * (.5 - x) * p[1] * (1. - p[1]) :
                                1.2 - 4.8 * std::abs(p[0] - .5);
    }
  }
  MappingFEField<2, 2, Vector<double>> moving(
    dofs, solution, fe.component_mask(FEValuesExtractors::Vector(0)));
  Parameters::BoundaryConditionsData bc;
  ParameterReader<2>                 param(bc);
  ParameterHandler                   prm;
  param.declare(prm);
  prm.parse_input_from_string(
    "subsection FiniteElements\nset Mesh position degree = 2\nend\n"
    "subsection Cahn Hilliard\nset interface thickness = 0.1\n"
    "subsection profile correction\nset enable profile correction = true\n"
    "set profile correction model = local timescale\nend\nend\n");
  param.read(prm);
  const auto fields =
    CHNSProfileGeometry::build(dofs, moving, solution, 2, param, true);
  AssertThrow(fields->overlap->size() > 0,
              ExcMessage("Missing interaction fixture"));
  for (const double x : {.4, .5, .6})
  {
    const auto value = fields->overlap->evaluate({x, 0.});
    AssertThrow(value.value > 0., ExcMessage("Inactive reflected penalty"));
    AssertThrow(
      std::abs(value.gradient.y) < 1e-10,
      ExcMessage(
        "Curved interior cells disabled reflection at a straight wall"));
  }
  deallog << "Q2 interior curvature preserves reflection at physical straight "
             "walls OK"
          << std::endl;
}
