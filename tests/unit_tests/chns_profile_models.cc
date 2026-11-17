#include <parameters.h>

#include "../tests.h"

using namespace dealii;

int main()
{
  initlog();
  deal_II_exceptions::disable_abort_on_exception();
  try
  {
    Parameters::CahnHilliard<2> ch;
    ParameterHandler            prm;
    ch.declare_parameters(prm);
    prm.parse_input_from_string(R"(
subsection Cahn Hilliard
  subsection profile correction
    set enable profile correction = true
    set profile correction model = constant
    set profile correction strength = 100
  end
end
)");
    ch.read_parameters(prm);
    AssertThrow(ch.enable_profile_correction &&
                  ch.profile_correction_strength == 100.,
                ExcMessage("Constant correction parameters were not read"));
    prm.parse_input_from_string(R"(
subsection Cahn Hilliard
  subsection profile correction
    set profile correction model = local timescale
    set profile correction timescale factor = 50
  end
end
)");
    ch.read_parameters(prm);
    AssertThrow(ch.local_profile_correction() &&
                  ch.profile_correction_timescale_factor == 50.,
                ExcMessage("Local correction parameters were not read"));
    // Invalid mobility must fail only when the local model is active.
    prm.parse_input_from_string(R"(
subsection Cahn Hilliard
  subsection mobility
    set model = degenerate
  end
end
)");
    bool rejected = false;
    try
    {
      ch.read_parameters(prm);
    }
    catch (const std::exception &)
    {
      rejected = true;
      prm.leave_subsection();
    }
    AssertThrow(rejected,
                ExcMessage("Local correction accepted variable mobility"));
    prm.parse_input_from_string(R"(
subsection Cahn Hilliard
  subsection profile correction
    set profile correction model = constant
  end
end
)");
    ch.read_parameters(prm);
    AssertThrow(
      ch.enable_profile_correction && ch.profile_correction_strength == 100.,
      ExcMessage("Constant correction lost variable-mobility support"));
    prm.parse_input_from_string(R"(
subsection Cahn Hilliard
  subsection profile correction
    set profile correction model = local timescale
    set enable profile correction = false
  end
end
)");
    ch.read_parameters(prm);
    AssertThrow(!ch.local_profile_correction(), ExcInternalError());
    deallog
      << "Nested constant/local timescale parameters and mobility validation OK"
      << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
