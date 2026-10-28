#include <time_handler.h>

#include "../tests.h"

int main()
{
  initlog();
  Parameters::TimeIntegration p{};
  ParameterHandler            prm;
  p.declare_parameters(prm);
  prm.enter_subsection("Time integration");
  prm.set("scheme", "BDF2");
  prm.set("bdf start method", "BDF1");
  prm.set("t_initial", "0");
  prm.set("t_end", "1");
  prm.set("dt", "0.1");
  prm.leave_subsection();
  p.read_parameters(prm);
  p.n_time_intervals = 1;
  TimeHandler time(p);
  time.bdf_coefficients = {15., -20., 5.};
  ConditionalOStream quiet(std::cout, false);
  time.advance(quiet);
  AssertThrow(time.bdf_coefficients[2] == 0.,
              ExcMessage(
                "BDF1 startup must clear the BDF2 history coefficient"));
  time.current_time_iteration             = 3;
  time.current_time_iteration_in_interval = 3;
  time.time_steps[0]                      = 0.1;
  time.time_steps[1]                      = 0.05;
  p.dt                                    = 0.2;
  time.update_parameters_after_restart(p);
  AssertThrow(time.time_steps[0] == 0.1 && time.current_dt == 0.2,
              ExcMessage("Restart must preserve the last accepted time step"));
  deallog << "BDF startup and restart history OK" << std::endl;
}
