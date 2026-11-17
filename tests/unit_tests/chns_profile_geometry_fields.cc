#include <chns_profile_geometry.h>

#include "../tests.h"

int main()
{
  initlog();
  constexpr double    eps = .1;
  std::vector<double> d{0.}, p{0.};
  for (double alpha : CHNSProfileWidth::levels)
  {
    d.push_back(1.2 * std::sqrt(2.) * eps * std::atanh(alpha));
    p.push_back(alpha);
  }
  d.push_back(12. * eps);
  p.push_back(1.);
  const auto width = CHNSProfileWidth::measure_ray(d, p, eps);
  AssertThrow(width.valid && std::abs(width.error_squared - .04) < 1e-14 &&
                width.confidence == 1.,
              dealii::ExcInternalError());
  // A disappearing tail does not create a false width-restoration signal.
  p.back()        = .999;
  p[p.size() - 2] = .999;
  AssertThrow(!CHNSProfileWidth::measure_ray(d, p, eps).valid,
              dealii::ExcMessage("Desaturated tail reported a valid width"));

  using namespace CHNSProfileOverlap;
  for (const bool confined : {false, true})
  {
    SmoothPenalty    penalty({{{.1, -.5}, {.1, .5}, .03, .2, confined}},
                          eps,
                          std::sqrt(2.));
    const auto       q  = penalty.evaluate({.13, .07});
    constexpr double h  = 1e-6;
    const auto       dx = (penalty.evaluate({.13 + h, .07}).value -
                     penalty.evaluate({.13 - h, .07}).value) /
                    (2 * h);
    const auto dy = (penalty.evaluate({.13, .07 + h}).value -
                     penalty.evaluate({.13, .07 - h}).value) /
                    (2 * h);
    AssertThrow(q.value > 0. && std::abs(dx - q.gradient.x) < 1e-7 &&
                  std::abs(dy - q.gradient.y) < 1e-7,
                dealii::ExcMessage("Kernel gradient"));
  }
  ReflectingBox box;
  box.lower            = {0., -1.};
  box.upper            = {1., 1.};
  box.reflect_lower[0] = true;
  SmoothPenalty reflected({{{.1, -.5}, {.1, .5}, .03}},
                          eps,
                          std::sqrt(2.),
                          box);
  AssertThrow(std::abs(reflected.evaluate({0., .1}).gradient.x) < 1e-12,
              dealii::ExcMessage("Natural boundary reflection"));
  dealii::deallog << "Width feedback, disappearing tails, kernel gradients and "
                     "boundary reflection OK"
                  << std::endl;
}
