
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <post_processing_handler.h>
#include <postprocessors_and_evaluators.h>

#include <filesystem>

template <int dim>
PostProcessingHandler<dim>::PostProcessingHandler(
  const ComponentOrdering                                 &ordering,
  const ParameterReader<dim>                              &param,
  const Triangulation<dim>                                &triangulation,
  const DoFHandler<dim>                                   &dof_handler,
  const std::vector<std::pair<std::string, unsigned int>> &fields_description)
  : ordering(ordering)
  , post_proc_param(param.postprocessing)
  , output_param(param.output)
  , physical_properties(param.physical_properties)
  , mms_param(param.mms_param)
  , fe_param(param.finite_elements)
  , triangulation(&triangulation, typeid(*this).name())
  , dof_handler(&dof_handler, typeid(*this).name())
  , mpi_communicator(dof_handler.get_mpi_communicator())
  , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  if (output_param.write_results || output_param.skin.write_results)
  {
    solution_names.clear();
    data_component_interpretation.clear();

    for (const auto &[name, n_comp] : fields_description)
      for (unsigned int d = 0; d < n_comp; ++d)
      {
        solution_names.push_back(name);
        data_component_interpretation.push_back(
          n_comp == 1 ?
            DataComponentInterpretation::component_is_scalar :
            DataComponentInterpretation::component_is_part_of_vector);
      }
  }

  this->attach_triangulation_and_dof_handler(triangulation, dof_handler);

  // Create the required DataPostProcessors
  {
    const auto &pp = param.postprocessing;
    using PP       = Parameters::PostProcessing;

    postprocessors.clear();
    if (ordering.has_variable(SolverInfo::VariableType::phase_tracer) &&
        ordering.has_variable(SolverInfo::VariableType::phase_potential) &&
        ordering.has_variable(SolverInfo::VariableType::pressure))
      postprocessors.emplace_back(
        std::make_unique<PostProcessingTools::AbelsPressurePostProcessor<dim>>(
          ordering));
    if (pp.density.enable &&
        pp.density.method == PP::Density::ComputationMethod::discontinuous)
      postprocessors.emplace_back(
        std::make_unique<PostProcessingTools::DensityPostProcessor<dim>>(
          ordering, param));
    if (pp.vorticity.enable and
        pp.vorticity.method == PP::Vorticity::ComputationMethod::discontinuous)
      postprocessors.emplace_back(
        std::make_unique<PostProcessingTools::VorticityPostProcessor<dim>>(
          ordering));
    if (pp.q_criterion.enable and
        pp.q_criterion.method ==
          PP::QCriterion::ComputationMethod::discontinuous)
      postprocessors.emplace_back(
        std::make_unique<PostProcessingTools::QCriterionPostProcessor<dim>>(
          ordering));
  }
}

template <int dim>
void PostProcessingHandler<dim>::attach_triangulation_and_dof_handler(
  const Triangulation<dim> &triangulation,
  const DoFHandler<dim>    &dof_handler)
{
  Assert(&dof_handler.get_triangulation() == &triangulation,
         ExcMessage("The provided triangulation does not match the one "
                    "attached to the provided DoFHandler"));

  // Assign triangulation and dof handler to this object
  this->triangulation = &triangulation;
  this->dof_handler   = &dof_handler;

  // Create new DataOuts
  if (output_param.write_results)
  {
    data_out = std::make_unique<DataOut<dim>>();
    data_out->attach_dof_handler(dof_handler);

    // Write high-order elements if needed
    if (fe_param.mapping_degree > 1)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out->set_flags(flags);
    }
  }

  if (output_param.skin.write_results)
  {
    // build_patches is not (yet) implemented for DataOutFaces in hp context,
    // but at this point the dof_handler might not yet be initialized.
    // The check is done in output_skin_fields instead.
    data_out_skin =
      std::make_unique<PostProcessingTools::DataOutFacesOnBoundary<dim>>(
        triangulation, output_param.skin.boundary_id);
    data_out_skin->attach_dof_handler(dof_handler);

    // Write high-order elements if needed
    if (fe_param.mapping_degree > 1)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out_skin->set_flags(flags);
    }
  }

  // Clear the stored cell-based vectors
  subdomains.reinit(0);
  slice_indices.reinit(0);
}

template <int dim>
void PostProcessingHandler<dim>::write_pvd(const PrefixData &prefix_data) const
{
  std::string suffix = "";
  prefix_data.append_to_prefix_or_suffix(output_param, true, suffix);
  suffix += ".pvd";

  if (mpi_rank == 0)
  {
    if (prefix_data.is_prerefinement_step)
    {
      if (!prerefinements_pseudotimes_and_names.empty())
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.output_prefix + suffix);
        DataOutBase::write_pvd_record(pvd_output,
                                      prerefinements_pseudotimes_and_names);
      }
      if (!prerefinements_pseudotimes_and_names_skin.empty())
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.skin.output_prefix + suffix);
        DataOutBase::write_pvd_record(
          pvd_output, prerefinements_pseudotimes_and_names_skin);
      }
    }
    else
    {
      if (output_param.write_results)
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.output_prefix + suffix);
        DataOutBase::write_pvd_record(pvd_output,
                                      visualization_times_and_names);
      }
      if (output_param.skin.write_results)
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.skin.output_prefix + suffix);
        DataOutBase::write_pvd_record(pvd_output,
                                      visualization_times_and_names_skin);
      }
    }
  }
}

template <int dim>
void PostProcessingHandler<dim>::create_slices()
{
  const std::string &dir = post_proc_param.slices.along_which_axis;

  AssertThrow(dir == "x" || dir == "y" || (dim == 3 && dir == "z"),
              ExcMessage(dim == 2 ?
                           "slicing_direction must be 'x' or 'y' in 2D." :
                           "slicing_direction must be 'x', 'y' or 'z' in 3D."));

  using SliceAxis      = PostProcessingTools::SliceAxis;
  const SliceAxis axis = (dir == "x" ? SliceAxis::x :
                          dir == "y" ? SliceAxis::y :
                                       SliceAxis::z);

  for (const auto id : post_proc_param.slices.boundary_ids)
    PostProcessingTools::set_slice_index_on_boundary<dim>(
      *triangulation, id, post_proc_param.slices.n_slices, axis);

  // Store slice indices as cell-based data.
  // If a face is on the sliced boundary, set its cell slice index to the
  // face user index.
  slice_indices.reinit(triangulation->n_active_cells());
  for (const auto &cell : triangulation->active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            std::binary_search(post_proc_param.slices.boundary_ids.begin(),
                               post_proc_param.slices.boundary_ids.end(),
                               face->boundary_id()))
        {
          slice_indices[cell->active_cell_index()] = face->user_index();
          break;
        }
}

template <int dim>
void PostProcessingHandler<dim>::clear()
{
  if (data_out)
    data_out->clear_data_vectors();
  if (data_out_skin)
    data_out_skin->clear_data_vectors();
  visualization_times_and_names.clear();
  visualization_times_and_names_skin.clear();
  subdomains.reinit(0);
  slice_indices.reinit(0);
}

template <int dim>
void PostProcessingHandler<dim>::add_force_to_table(
  const Tensor<1, dim>    &forces,
  const TimeHandler       &time_handler,
  TableHandler            &force_table,
  const types::boundary_id boundary_id,
  const unsigned int       i_slice)
{
  // Write forces to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  force_table.add_value("time", time_handler.current_time);
  force_table.add_value("boundary", boundary_id);
  if (i_slice != numbers::invalid_unsigned_int)
    force_table.add_value("slice", i_slice);
  for (unsigned int d = 0; d < dim; ++d)
  {
    force_table.add_value("F" + dim_str[d], forces[d]);
    force_table.set_precision("F" + dim_str[d],
                              post_proc_param.forces.precision);
    force_table.set_scientific("F" + dim_str[d], true);
  }
}

template <int dim>
void PostProcessingHandler<dim>::add_position_to_table(
  const Tensor<1, dim>    &center_position,
  const TimeHandler       &time_handler,
  TableHandler            &table,
  const types::boundary_id boundary_id)
{
  // Write position to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  table.add_value("time", time_handler.current_time);
  table.add_value("boundary", boundary_id);
  for (unsigned int d = 0; d < dim; ++d)
  {
    table.add_value(dim_str[d], center_position[d]);
    table.set_precision(dim_str[d],
                        post_proc_param.structure_position.precision);
    table.set_scientific(dim_str[d], true);
  }
}

template <int dim>
template <typename DataType>
void PostProcessingHandler<dim>::add_multiphase_data_to_table(
  const std::array<DataType, 2>                        &data_for_phases,
  const TimeHandler                                    &time_handler,
  TableHandler                                         &table,
  const Parameters::PostProcessing::PostProcessingFile &pp_param)
{
  if constexpr (std::is_same_v<DataType, Tensor<1, dim>>)
  {
    std::vector<std::string> dim_str = {"x", "y", "z"};
    table.add_value("time", time_handler.current_time);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int d = 0; d < dim; ++d)
      {
        std::string key = "phase" + std::to_string(i) + "_" + dim_str[d];
        table.add_value(key, data_for_phases[i][d]);
        table.set_precision(key, pp_param.precision);
        table.set_scientific(key, true);
      }
  }
  else
  {
    table.add_value("time", time_handler.current_time);
    for (unsigned int i = 0; i < 2; ++i)
    {
      std::string key = "phase" + std::to_string(i);
      table.add_value(key, data_for_phases[i]);
      table.set_precision(key, pp_param.precision);
      table.set_scientific(key, true);
    }
  }
}

// Explicit instantiations for dim = 2,3, two phases and double/Tensor<1, dim>
template void PostProcessingHandler<2>::add_multiphase_data_to_table(
  const std::array<Tensor<1, 2>, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);
template void PostProcessingHandler<3>::add_multiphase_data_to_table(
  const std::array<Tensor<1, 3>, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);
template void PostProcessingHandler<2>::add_multiphase_data_to_table(
  const std::array<double, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);
template void PostProcessingHandler<3>::add_multiphase_data_to_table(
  const std::array<double, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);

template <int dim>
void PostProcessingHandler<dim>::write_table(
  std::ostream                                         &out,
  const TableHandler                                   &table,
  const Parameters::PostProcessing::PostProcessingFile &postproc_file) const
{
  if (mpi_rank == 0)
  {
    out << std::scientific << std::setprecision(postproc_file.precision);
    table.write_text(out);
  }
}

template <int dim>
void PostProcessingHandler<dim>::write_forces(std::ostream &out) const
{
  write_table(out, forces_table, post_proc_param.forces);
}

template <int dim>
void PostProcessingHandler<dim>::write_structure_mean_position(
  std::ostream &out) const
{
  write_table(out,
              structure_mean_position_table,
              post_proc_param.structure_position);
}

template <int dim>
std::unique_ptr<PostProcessingTools::PostprocessorAtDofBase<dim>>
PostProcessingHandler<dim>::create_field_postprocessor(
  const PostProcessingTools::PostprocessorAtDofTypes type,
  const ParameterReader<dim>                        &param,
  const Mapping<dim>                                &mapping,
  const Quadrature<dim>                             &cell_quadrature,
  const bool                                         with_moving_mesh)
{
  using namespace PostProcessingTools;
  using Vorticity  = Parameters::PostProcessing::Vorticity;
  using QCriterion = Parameters::PostProcessing::QCriterion;
  using Field      = Parameters::PostProcessing::PostProcessingField;

  switch (type)
  {
    case PostprocessorAtDofTypes::vorticity:
      switch (param.postprocessing.vorticity.method)
      {
        case Vorticity::ComputationMethod::discontinuous:
          // Nothing to do: the DataPostprocessor was created in the
          // constructor.
          return nullptr;
        case Vorticity::ComputationMethod::l2_projection:
          return std::make_unique<
            FieldPostprocessorGenerator<dim, VorticityEvaluator, L2Projection>>(
            param,
            ordering,
            mapping,
            *dof_handler,
            cell_quadrature,
            with_moving_mesh);
        case Vorticity::ComputationMethod::weighted_average:
          return std::make_unique<
            FieldPostprocessorGenerator<dim,
                                        VorticityEvaluator,
                                        WeightedAverage>>(param,
                                                          ordering,
                                                          mapping,
                                                          *dof_handler,
                                                          cell_quadrature,
                                                          with_moving_mesh);
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    case PostprocessorAtDofTypes::q_criterion:
      switch (param.postprocessing.q_criterion.method)
      {
        case QCriterion::ComputationMethod::discontinuous:
          // Nothing to do: the DataPostprocessor was created in the
          // constructor.
          return nullptr;
        case QCriterion::ComputationMethod::l2_projection:
          return std::make_unique<
            FieldPostprocessorGenerator<dim,
                                        QCriterionEvaluator,
                                        L2Projection>>(param,
                                                       ordering,
                                                       mapping,
                                                       *dof_handler,
                                                       cell_quadrature,
                                                       with_moving_mesh);
        case QCriterion::ComputationMethod::weighted_average:
          return std::make_unique<
            FieldPostprocessorGenerator<dim,
                                        QCriterionEvaluator,
                                        WeightedAverage>>(param,
                                                          ordering,
                                                          mapping,
                                                          *dof_handler,
                                                          cell_quadrature,
                                                          with_moving_mesh);
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    case PostprocessorAtDofTypes::mesh_velocity:
      return std::make_unique<MeshVelocityPostprocessor<dim>>(
        ordering, param, mapping, *dof_handler, cell_quadrature);
    case PostprocessorAtDofTypes::density:
      switch (param.postprocessing.density.method)
      {
        case Field::ComputationMethod::l2_projection:
          return std::make_unique<
            FieldPostprocessorGenerator<dim, DensityEvaluator, L2Projection>>(
            param,
            ordering,
            mapping,
            *dof_handler,
            cell_quadrature,
            with_moving_mesh);
        case Field::ComputationMethod::weighted_average:
          return std::make_unique<FieldPostprocessorGenerator<dim,
                                                              DensityEvaluator,
                                                              WeightedAverage>>(
            param,
            ordering,
            mapping,
            *dof_handler,
            cell_quadrature,
            with_moving_mesh);
        case Field::ComputationMethod::discontinuous:
          // Direct visualization is handled by the DataPostprocessor created
          // in the constructor; no projected field is needed.
          return nullptr;
      }
      DEAL_II_ASSERT_UNREACHABLE();
      return nullptr;
    case PostprocessorAtDofTypes::mobility:
      switch (param.postprocessing.mobility.method)
      {
        case Field::ComputationMethod::l2_projection:
          return std::make_unique<
            FieldPostprocessorGenerator<dim, MobilityEvaluator, L2Projection>>(
            param,
            ordering,
            mapping,
            *dof_handler,
            cell_quadrature,
            with_moving_mesh);
        case Field::ComputationMethod::weighted_average:
          return std::make_unique<FieldPostprocessorGenerator<dim,
                                                              MobilityEvaluator,
                                                              WeightedAverage>>(
            param,
            ordering,
            mapping,
            *dof_handler,
            cell_quadrature,
            with_moving_mesh);
        case Field::ComputationMethod::discontinuous:
          AssertThrow(false,
                      ExcMessage("Mobility postprocessing supports 'l2 "
                                 "projection' and 'weighted average'."));
          return nullptr;
      }
      DEAL_II_ASSERT_UNREACHABLE();
      return nullptr;
    default:
      DEAL_II_NOT_IMPLEMENTED();
  }
  DEAL_II_ASSERT_UNREACHABLE();
  return nullptr;
}

template <int dim>
void PostProcessingHandler<dim>::create_field_postprocessors(
  const ParameterReader<dim> &param,
  const Mapping<dim>         &mapping,
  const Quadrature<dim>      &cell_quadrature,
  const bool                  with_moving_mesh)
{
  for (const auto &[type, postprocessor_param_ptr] :
       param.postprocessing.field_postprocessors)
    if (postprocessor_param_ptr->enable)
      field_postprocessors[type] = create_field_postprocessor(
        type, param, mapping, cell_quadrature, with_moving_mesh);
}

template <int dim>
void PostProcessingHandler<dim>::process_chns_energy(
  const CHNSEnergy::Integrals &data,
  const TimeHandler           &time)
{
  using I       = CHNSEnergy::Integrals;
  const auto &v = data.values;
  if (chns_energy_last_iteration == time.current_time_iteration)
    return;
  const bool first = !chns_energy_initialized;
  const bool power_available =
    time.is_steady() ||
    (time.current_time_iteration > 0 &&
     time.current_time_iteration != time.time_iteration_at_last_restart);
  const auto power = [&](double value) {
    return power_available ? value : std::numeric_limits<double>::quiet_NaN();
  };
  if (first)
  {
    chns_energy_initialized    = true;
    chns_energy_reference_time = chns_energy_last_time = time.current_time;
    chns_initial_mass                                  = v[I::AffineMass];
    chns_initial_phase                                 = v[I::Phase];
    chns_initial_energy = chns_previous_energy = data.total_energy();
  }
  const double dt = time.current_time - chns_energy_last_time;
  for (unsigned int k = 0; k < 3; ++k)
    chns_stabilization_work[k] += dt * v[I::SUPGPower + k];
  chns_energy_last_time      = time.current_time;
  chns_energy_last_iteration = time.current_time_iteration;
  const std::array<const Parameters::PostProcessing::PostProcessingFile *, 4>
              parameters{{&post_proc_param.chns_conservation,
                          &post_proc_param.chns_energy,
                          &post_proc_param.chns_phase_energy,
                          &post_proc_param.chns_stabilization_energy}};
  std::string error;
  if (mpi_rank == 0)
    try
    {
      for (unsigned int k = 0; k < parameters.size(); ++k)
      {
        const auto &param = *parameters[k];
        if (!param.enable)
          continue;
        auto      &table = chns_energy_tables[k];
        const auto add   = [&](const std::string &name, double value) {
          table.add_value(name, value);
          table.set_precision(name, param.precision);
          table.set_scientific(name, true);
        };
        add("time", time.current_time);
        add("reference_time", chns_energy_reference_time);
        table.add_value("iteration", time.current_time_iteration);
        if (k == 0)
        {
          add("domain_volume", v[I::Volume]);
          add("phase_integral", v[I::Phase]);
          add("mass_affine", v[I::AffineMass]);
          add("mass_positive", v[I::PositiveMass]);
          add("phase_change_from_reference", v[I::Phase] - chns_initial_phase);
          add("mass_change_from_reference",
              v[I::AffineMass] - chns_initial_mass);
          add("divergence_l2", std::sqrt(v[I::DivergenceSquared]));
          add("phase_overshoot_volume", v[I::PhaseOvershootVolume]);
          add("density_extension_l1", v[I::DensityExtensionL1]);
          add("outward_boundary_phase_flux", power(v[I::BoundaryPhaseFlux]));
        }
        if (k == 1 || k == 2)
        {
          const std::array<std::string, 4> names{{"kinetic_energy",
                                                  "bulk_energy",
                                                  "gradient_energy",
                                                  "gravitational_energy"}};
          const unsigned int               count = k == 1 ? 1 : 2;
          for (unsigned int phase = 0; phase < count; ++phase)
          {
            const unsigned int offset =
              k == 1 ? static_cast<unsigned int>(I::Kinetic) :
                       I::Phase0Kinetic + 4 * phase;
            const std::string suffix =
              k == 1 ? "" : "_phase_" + std::to_string(phase);
            double total = 0.;
            for (unsigned int e = 0; e < 4; ++e)
            {
              add(names[e] + suffix, v[offset + e]);
              total += v[offset + e];
            }
            add("total_energy" + suffix, total);
          }
        }
        if (k == 1)
        {
          add("energy_change_from_reference",
              data.total_energy() - chns_initial_energy);
          add("energy_step_change", data.total_energy() - chns_previous_energy);
          add("viscous_dissipation", v[I::Viscous]);
          add("diffusive_dissipation", v[I::Diffusive]);
          add("profile_correction_power", v[I::CorrectionPower]);
          add("body_force_power", v[I::BodyPower]);
          add("source_power", power(v[I::SourcePower]));
          add("temporal_power", power(v[I::TemporalPower]));
          add("transport_power", power(v[I::TransportPower]));
          add("stabilization_power", power(data.stabilization_power()));
          // Volume identity for kinetic + free energy, before adding boundary
          // work. The gravitational energy is reported separately above.
          const double residual = v[I::TemporalPower] + v[I::TransportPower] +
                                  v[I::Viscous] + v[I::Diffusive] +
                                  v[I::CorrectionPower] + v[I::SourcePower] -
                                  v[I::BodyPower] + data.stabilization_power();
          add("volume_free_kinetic_power_residual", power(residual));
          add("outward_boundary_kinetic_power", power(v[I::BoundaryKineticPower]));
          add("outward_boundary_phase_power", power(v[I::BoundaryPhasePower]));
          add("boundary_traction_power", power(v[I::BoundaryTractionPower]));
          add("boundary_reaction_power", power(v[I::BoundaryReactionPower]));
          // Reactions give the discrete work of prescribed velocity components.
          // Traction above is a separate stress-trace estimate, not an extra
          // force. The balance is for kinetic + free energy, excluding gravity.
          add("free_kinetic_power_residual",
              power(residual + v[I::BoundaryKineticPower] +
                    v[I::BoundaryPhasePower] - v[I::BoundaryReactionPower]));
        }
        if (k == 3)
        {
          const std::array<std::string, 3> names{
            {"momentum_supg", "pspg", "tracer_supg"}};
          for (unsigned int j = 0; j < 3; ++j)
          {
            add(names[j] + "_power", power(v[I::SUPGPower + j]));
            add(names[j] + "_work", chns_stabilization_work[j]);
          }
          add("total_power", power(data.stabilization_power()));
          add("total_work",
              chns_stabilization_work[0] + chns_stabilization_work[1] +
                chns_stabilization_work[2]);
        }
        if (param.verbosity == Parameters::Verbosity::verbose)
          std::cout << param.output_prefix << " at t=" << time.current_time
                    << ": energy=" << data.total_energy()
                    << ", stabilization power=" << data.stabilization_power()
                    << std::endl;
        if (should_output_postprocessing(time, param))
        {
          std::filesystem::create_directories(output_param.output_dir);
          const auto filename =
            output_param.output_dir + param.output_prefix + ".txt";
          std::ofstream out(filename);
          AssertThrow(out.good(), ExcMessage("Cannot write " + filename));
          write_table(out, table, param);
          AssertThrow(out.good(), ExcMessage("Failed to write " + filename));
        }
      }
    }
    catch (const std::exception &exception)
    {
      error = exception.what();
    }
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(), ExcMessage(error));
  chns_previous_energy = data.total_energy();
}

template <int dim>
void PostProcessingHandler<dim>::save_chns_energy(
  const std::string &prefix) const
{
  // Always replace the sidecar, including an inactive diagnostic state.
  // Otherwise reusing a checkpoint prefix could retain an older reference.
  std::string error;
  if (mpi_rank == 0)
    try
    {
      std::ofstream file(prefix + ".chns_energy", std::ios::binary);
      AssertThrow(file.good(),
                  ExcMessage("Cannot write CHNS energy checkpoint"));
      file << "FEZ_ABELS_ENERGY_V1\n";
      boost::archive::binary_oarchive archive(file);
      archive << chns_energy_initialized << chns_energy_last_iteration
              << chns_energy_reference_time << chns_energy_last_time
              << chns_initial_mass << chns_initial_phase << chns_initial_energy
              << chns_previous_energy;
      for (const auto work : chns_stabilization_work)
        archive << work;
      for (const auto &table : chns_energy_tables)
        archive << table;
    }
    catch (const std::exception &exception)
    {
      error = exception.what();
    }
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(), ExcMessage(error));
}

template <int dim>
void PostProcessingHandler<dim>::load_chns_energy(const std::string &prefix)
{
  std::string error;
  if (mpi_rank == 0)
    try
    {
      std::ifstream file(prefix + ".chns_energy", std::ios::binary);
      // Older checkpoints start a new, explicitly dated diagnostic reference.
      if (file)
      {
        std::string version;
        std::getline(file, version);
        AssertThrow(version == "FEZ_ABELS_ENERGY_V1",
                    ExcMessage("Unknown CHNS energy checkpoint format"));
        boost::archive::binary_iarchive archive(file);
        archive >> chns_energy_initialized >> chns_energy_last_iteration >>
          chns_energy_reference_time >> chns_energy_last_time >>
          chns_initial_mass >> chns_initial_phase >> chns_initial_energy >>
          chns_previous_energy;
        for (auto &work : chns_stabilization_work)
          archive >> work;
        for (auto &table : chns_energy_tables)
          archive >> table;
      }
    }
    catch (const std::exception &exception)
    {
      error = exception.what();
    }
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(), ExcMessage(error));
  chns_energy_initialized =
    Utilities::MPI::broadcast(mpi_communicator, chns_energy_initialized, 0);
  chns_energy_last_iteration =
    Utilities::MPI::broadcast(mpi_communicator, chns_energy_last_iteration, 0);
  std::vector<double> state = {chns_energy_reference_time,
                               chns_energy_last_time,
                               chns_initial_mass,
                               chns_initial_phase,
                               chns_initial_energy,
                               chns_previous_energy,
                               chns_stabilization_work[0],
                               chns_stabilization_work[1],
                               chns_stabilization_work[2]};
  state = Utilities::MPI::broadcast(mpi_communicator, state, 0);
  chns_energy_reference_time = state[0];
  chns_energy_last_time      = state[1];
  chns_initial_mass          = state[2];
  chns_initial_phase         = state[3];
  chns_initial_energy        = state[4];
  chns_previous_energy       = state[5];
  for (unsigned int k = 0; k < 3; ++k)
    chns_stabilization_work[k] = state[6 + k];
}

template class PostProcessingHandler<2>;
template class PostProcessingHandler<3>;
