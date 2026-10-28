#include <chns_energy.h>

#include "../tests.h"

int main()
{
  initlog();
  using namespace CHNSEnergy;
  Tensor<1, 2> u, old_u, older_u;
  u[0]       = 0.7;
  u[1]       = -0.2;
  old_u[0]   = -0.1;
  old_u[1]   = 0.4;
  older_u[0] = 0.2;
  older_u[1] = 0.3;
  const std::array<double, 2>       phases{{-0.3, 0.1}};
  const std::array<Tensor<1, 2>, 2> velocities{{old_u, older_u}};
  for (const auto &coefficients :
       {std::vector<double>{10., -10.}, std::vector<double>{15., -20., 5.}})
    for (const double phase : {-1.1, 0.2, 1.1})
    {
      const auto t =
        evaluate_time(phase, u, 10., 1., coefficients, phases, velocities);
      const double h = 1.e-6;
      const auto   plus =
        evaluate_time(phase + h, u, 10., 1., coefficients, phases, velocities);
      const auto minus =
        evaluate_time(phase - h, u, 10., 1., coefficients, phases, velocities);
      AssertThrow(((plus.inertia - minus.inertia) / (2 * h) -
                   t.inertia_phase_derivative)
                      .norm() < 1.e-7,
                  ExcMessage("Incorrect phase derivative of inertia"));
      AssertThrow(std::abs(
                    (plus.potential_derivative - minus.potential_derivative) /
                      (2 * h) -
                    t.potential_second_derivative) < 1.e-8,
                  ExcInternalError());
      auto up = u;
      up[0] += h;
      const auto tp =
        evaluate_time(phase, up, 10., 1., coefficients, phases, velocities);
      AssertThrow(std::abs((tp.inertia[0] - t.inertia[0]) / h -
                           t.inertia_velocity_derivative) < 1.e-7,
                  ExcMessage("Incorrect velocity derivative of inertia"));
      if (coefficients.size() == 2)
      {
        const double old_rho = positive_density(phases[0], 10., 1.);
        const double energy_increment =
          0.5 * (t.density * u.norm_square() - old_rho * old_u.norm_square());
        const double loss = 0.5 * old_rho * (u - old_u).norm_square();
        AssertThrow(std::abs(t.inertia * u / coefficients[0] -
                             energy_increment - loss) < 1.e-12,
                    ExcMessage("BDF1 kinetic identity"));
        const auto F = [](double a) {
          return 0.25 * (a * a - 1) * (a * a - 1);
        };
        AssertThrow(std::abs(t.potential_derivative * (phase - phases[0]) -
                             F(phase) + F(phases[0])) < 1.e-12,
                    ExcMessage("Discrete potential chain rule"));
      }
      else
      {
        const auto a = std::sqrt(t.density) * u;
        const auto b = std::sqrt(positive_density(phases[0], 10., 1.)) * old_u;
        const auto c =
          std::sqrt(positive_density(phases[1], 10., 1.)) * older_u;
        const double difference =
          0.25 * (a.norm_square() + (2 * a - b).norm_square() -
                  b.norm_square() - (2 * b - c).norm_square());
        AssertThrow(std::abs(0.1 * t.inertia * u - difference -
                             0.25 * (a - 2 * b + c).norm_square()) < 1.e-12,
                    ExcMessage("BDF2 kinetic G identity"));
      }
    }
  deallog << "Temporal identities and derivatives OK" << std::endl;
}
