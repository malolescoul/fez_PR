#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_handler.h>
#include <postprocessors_and_evaluators.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "../tests.h"

std::string read_file(const std::string &name)
{
  std::ifstream in(name);
  AssertThrow(in.good(), ExcMessage("Missing output: " + name));
  return {std::istreambuf_iterator<char>(in), {}};
}

std::vector<std::vector<double>> read_rows(const std::string &text,
                                           const unsigned int columns)
{
  std::istringstream               table(text);
  std::vector<std::vector<double>> rows;
  std::string                      line;
  while (std::getline(table, line))
  {
    std::istringstream  data(line);
    double              x;
    std::vector<double> row;
    while (data >> x)
      row.push_back(x);
    if (row.size() == columns)
      rows.push_back(row);
  }
  return rows;
}

template <int dim>
class Fields : public Function<dim>
{
public:
  Fields(const double offset = 1.)
    : Function<dim>(dim + 3)
    , offset(offset)
  {}
  double value(const Point<dim> &p, const unsigned int c) const override
  {
    if (c == dim + 1)
      return offset +
             p[0]; // Deliberately exceed one: assembly does not clamp phi.
    if (c == dim + 2)
      return 2. * p[0] - 3. * p[1] + (dim == 3 ? 4. * p[dim - 1] : 0.);
    return 0.;
  }

private:
  const double offset;
};

template <int dim>
void check_force()
{
  const unsigned rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ComponentOrderingCHNS<dim, false>                          ordering;
  PostProcessingTools::SurfaceTensionForcePostProcessor<dim> pp(ordering);
  bool                                                       rejected = false;
  try
  {
    PostProcessingTools::SurfaceTensionForcePostProcessor<dim> invalid{
      ComponentOrderingNS<dim>()};
  }
  catch (const ExceptionBase &)
  {
    rejected = true;
  }
  AssertThrow(rejected, ExcMessage("Non-CHNS force output must be rejected"));
  DataPostprocessorInputs::Vector<dim> input;
  input.solution_values.assign(3, Vector<double>(dim + 3));
  input.solution_gradients.resize(3, std::vector<Tensor<1, dim>>(dim + 3));
  const double phases[]   = {-1.25, 0., 1.5};
  const double gradient[] = {2., -3., 4.};
  for (unsigned q = 0; q < 3; ++q)
  {
    input.solution_values[q][dim + 1] = phases[q];
    for (unsigned d = 0; d < dim; ++d)
      input.solution_gradients[q][dim + 2][d] = gradient[d];
  }
  std::vector<Vector<double>> values(3, Vector<double>(dim));
  pp.evaluate_vector_field(input, values);
  for (unsigned d = 0; d < dim; ++d)
  {
    AssertThrow(values[0][d] == 1.25 * gradient[d] && values[1][d] == 0. &&
                  values[2][d] == -1.5 * gradient[d],
                ExcInternalError());
    AssertThrow(pp.get_data_component_interpretation()[d] ==
                  DataComponentInterpretation::component_is_part_of_vector,
                ExcInternalError());
  }
  for (auto &gradients : input.solution_gradients)
    gradients[dim + 2] = {};
  pp.evaluate_vector_field(input, values);
  for (const auto &value : values)
    AssertThrow(value.l2_norm() == 0., ExcInternalError());

  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, 0., 2., true);
  tria.refine_global(1);
  FESystem<dim>   fe(FE_Q<dim>(1), dim + 3);
  DoFHandler<dim> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<dim>    mapping;
  QGauss<dim>       quadrature(2);
  LA::ParVectorType owned, solution;
  owned.reinit(dofs.locally_owned_dofs(), MPI_COMM_WORLD);
  solution.reinit(dofs.locally_owned_dofs(),
                  DoFTools::extract_locally_relevant_dofs(dofs),
                  MPI_COMM_WORLD);
  VectorTools::interpolate(mapping, dofs, Fields<dim>(), owned);
  solution = owned;

  Parameters::BoundaryConditionsData bc{};
  ParameterReader<dim>               param(bc);
  ParameterHandler                   prm;
  param.postprocessing.declare_parameters(prm);
  param.output.declare_parameters(prm);
  param.time_integration.declare_parameters(prm);
  param.postprocessing.read_parameters(prm);
  AssertThrow(!param.postprocessing.surface_tension_force.enable,
              ExcMessage("Force output must be disabled by default"));
  std::istringstream settings(
    "subsection Postprocessing\nsubsection surface tension force\n"
    "set enable = true\nset write results = true\nset precision = 12\n"
    "set verbosity = quiet\nend\nend\n");
  prm.parse_input(settings);
  param.postprocessing.read_parameters(prm);
  param.postprocessing.chns_energy.enable    = true;
  param.postprocessing.chns_energy.verbosity = Parameters::Verbosity::quiet;
  param.output.read_parameters(prm);
  param.time_integration.read_parameters(prm);
  param.time_integration.scheme = Parameters::TimeIntegration::Scheme::BDF1;
  param.time_integration.t_initial = 0.;
  param.time_integration.t_end = 1.;
  param.time_integration.n_time_intervals          = 1;
  param.time_integration.n_steady_adaptation_steps = 0;
  param.finite_elements.mapping_degree             = 1;
  param.output.output_dir     = "surface_force_" + std::to_string(dim) + "d/";
  param.output.output_prefix  = "job_A";
  param.output.write_results  = true;
  param.output.n_subdivisions = 1;
  if (rank == 0)
    std::filesystem::create_directories(param.output.output_dir);
  MPI_Barrier(MPI_COMM_WORLD);
  TimeHandler time(param.time_integration);
  const std::vector<std::pair<std::string, unsigned>> names{{"velocity", dim},
                                                            {"pressure", 1},
                                                            {"phase", 1},
                                                            {"potential", 1}};
  PostProcessingHandler<dim> handler(ordering, param, tria, dofs, names);
  handler.compute_surface_tension_force(
    dofs, mapping, quadrature, solution, time);
  handler.process_chns_energy(CHNSEnergy::Integrals{}, time);
  handler.output_fields(mapping, solution, time);
  const auto file_A =
    param.output.output_dir + "job_A_surface_tension_force.txt";
  std::string original;
  std::string original_energy;
  if (rank == 0)
  {
    original = read_file(file_A);
    original_energy =
      read_file(param.output.output_dir + "job_A_chns_energy.txt");
    const auto rows = read_rows(original, dim + 4);
    AssertThrow(rows.size() == 1, ExcMessage("Missing force table row"));
    const auto  &row    = rows.back();
    const double volume = std::pow(2., dim);
    for (unsigned d = 0; d < dim; ++d)
      AssertThrow(std::abs(row[d + 2] + 2. * volume * gradient[d]) < 1e-9,
                  ExcMessage("Incorrect global capillary force integral"));
    const double magnitude = 2. * volume * std::sqrt(dim == 2 ? 13. : 29.);
    AssertThrow(std::abs(row[dim + 2] - magnitude) < 1e-8 &&
                  std::abs(row[dim + 3] - magnitude) < 1e-8,
                ExcMessage("Incorrect force norms"));
    const auto pvtu = read_file(param.output.output_dir + "job_A_00.pvtu");
    AssertThrow(pvtu.find("surface_tension_force") != std::string::npos,
                ExcMessage("Missing ParaView force vector"));
  }
  param.output.output_prefix = "job_B";
  VectorTools::interpolate(mapping, dofs, Fields<dim>(-1.), owned);
  solution = owned;
  PostProcessingHandler<dim> other(ordering, param, tria, dofs, names);
  other.compute_surface_tension_force(
    dofs, mapping, quadrature, solution, time);
  other.process_chns_energy(CHNSEnergy::Integrals{}, time);
  if (rank == 0)
  {
    AssertThrow(read_file(file_A) == original,
                ExcMessage("Another job overwrote TXT"));
    AssertThrow(read_file(param.output.output_dir + "job_A_chns_energy.txt") ==
                  original_energy,
                ExcMessage("Another job overwrote energy TXT"));
    read_file(param.output.output_dir + "job_B_chns_energy.txt");
    const auto rows = read_rows(read_file(param.output.output_dir +
                                          "job_B_surface_tension_force.txt"),
                                dim + 4);
    AssertThrow(rows.size() == 1, ExcInternalError());
    const auto &row = rows.back();
    for (unsigned d = 0; d <= dim; ++d)
      AssertThrow(std::abs(row[d + 2]) < 1e-10,
                  ExcMessage("Forces must cancel"));
    const double magnitude =
      std::pow(2., dim - 1) * std::sqrt(dim == 2 ? 13. : 29.);
    AssertThrow(std::abs(row[dim + 3] - magnitude) < 1e-8,
                ExcMessage("Integral of magnitude must survive cancellation"));
  }
  time.current_time           = .05;
  time.current_time_iteration = 1;
  other.compute_surface_tension_force(
    dofs, mapping, quadrature, solution, time);
  other.save_chns_energy(param.output.output_dir + "checkpoint");
  PostProcessingHandler<dim> resumed(ordering, param, tria, dofs, names);
  resumed.load_chns_energy(param.output.output_dir + "checkpoint");
  resumed.compute_surface_tension_force(
    dofs, mapping, quadrature, solution, time);
  time.current_time           = .1;
  time.current_time_iteration = 2;
  resumed.compute_surface_tension_force(
    dofs, mapping, quadrature, solution, time);
  if (rank == 0)
  {
    const auto rows = read_rows(read_file(param.output.output_dir +
                                          "job_B_surface_tension_force.txt"),
                                dim + 4);
    AssertThrow(
      rows.size() == 3 && rows[0][0] == 0. && rows[1][0] == .05 &&
        rows[2][0] == .1,
      ExcMessage(
        "Force TXT restart lost history or duplicated the checkpoint"));
  }
  time.current_time                                        = 0.;
  time.current_time_iteration                              = 0;
  param.output.output_prefix                               = "disabled";
  param.postprocessing.surface_tension_force.write_results = false;
  PostProcessingHandler<dim> no_text(ordering, param, tria, dofs, names);
  no_text.compute_surface_tension_force(
    dofs, mapping, quadrature, solution, time);
  if (rank == 0)
    AssertThrow(!std::filesystem::exists(param.output.output_dir +
                                         "disabled_surface_tension_force.txt"),
                ExcInternalError());
  param.postprocessing.surface_tension_force.enable = false;
  PostProcessingHandler<dim> disabled(ordering, param, tria, dofs, names);
  disabled.output_fields(mapping, solution, time);
  if (rank == 0)
    AssertThrow(read_file(param.output.output_dir + "disabled_00.pvtu")
                    .find("surface_tension_force") == std::string::npos,
                ExcInternalError());

  // A restart must preserve the past, remove future records, and replace the
  // old record at the checkpoint when that timestep is exported again.
  param.output.output_prefix      = "restart";
  param.output.skin.write_results = true;
  param.output.skin.output_prefix = "restart_skin";
  const auto pvd                  = param.output.output_dir + "restart.pvd";
  if (rank == 0)
  {
    std::ofstream out(pvd);
    DataOutBase::write_pvd_record(
      out, {{0., "past.pvtu"}, {.1, "checkpoint.pvtu"}, {.2, "future.pvtu"}});
    std::ofstream skin(param.output.output_dir + "restart_skin.pvd");
    DataOutBase::write_pvd_record(skin,
                                  {{0., "skin_past.pvtu"},
                                   {.1, "skin_checkpoint.pvtu"},
                                   {.2, "skin_future.pvtu"}});
  }
  MPI_Barrier(MPI_COMM_WORLD);
  PostProcessingHandler<dim> restarted(ordering, param, tria, dofs, names);
  restarted.restore_pvd_entries_until_time(.1);
  time.current_time           = .1;
  time.current_time_iteration = 1;
  restarted.output_fields(mapping, solution, time);
  restarted.write_pvd();
  if (rank == 0)
  {
    const auto text = read_file(pvd);
    AssertThrow(text.find("past.pvtu") != std::string::npos &&
                  text.find("future.pvtu") == std::string::npos &&
                  text.find("checkpoint.pvtu") == std::string::npos &&
                  text.find("restart_01.pvtu") != std::string::npos,
                ExcMessage(
                  "PVD restart lost history or retained stale records"));
    const auto skin = read_file(param.output.output_dir + "restart_skin.pvd");
    AssertThrow(skin.find("skin_past.pvtu") != std::string::npos &&
                  skin.find("skin_future.pvtu") == std::string::npos &&
                  skin.find("skin_checkpoint.pvtu") == std::string::npos,
                ExcMessage(
                  "Skin PVD restart lost history or retained stale records"));
  }
  // The deliberate epsilon offset at a mesh-transfer boundary represents two
  // distinct solutions, not a duplicate checkpoint record.
  time.current_time = .25;
  time.current_time_iteration = 2;
  restarted.output_fields(mapping, solution, time);
  param.output.fixed_point.show_solution_transfer = true;
  typename PostProcessingHandler<dim>::PrefixData transfer;
  transfer.is_time_subinterval = true;
  transfer.interval_index = 1;
  time.current_time_iteration_in_interval = 0;
  restarted.output_fields(mapping, solution, time, transfer);
  if (rank == 0)
  {
    const auto text = read_file(pvd);
    AssertThrow(text.find("restart_02.pvtu") != std::string::npos &&
                  text.find("restart_int_1_02.pvtu") != std::string::npos,
                ExcMessage("PVD deduplication erased the pre-transfer solution"));
    const auto skin = read_file(param.output.output_dir + "restart_skin.pvd");
    AssertThrow(skin.find("restart_restart_skin_02.pvtu") != std::string::npos &&
                  skin.find("restart_restart_skin_int_1_02.pvtu") != std::string::npos,
                ExcMessage("Skin PVD deduplication erased the pre-transfer solution"));
    deallog << dim << "D force, TXT isolation and PVD restart OK" << std::endl;
  }
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  check_force<2>();
  check_force<3>();
}
