#include <chns_profile_correction_cutoff.h>

#include "../tests.h"

// Pointwise indicators of the interaction cutoff of the profile correction.
// Planar profiles are written along x, with phi, its gradient and its Hessian
// given in closed form.

struct Profile1D
{
  double phi, dphi, ddphi;
};

// tanh(x / (sqrt(2) w)) of width w.
Profile1D tanh_profile(const double x, const double w)
{
  const double phi = std::tanh(x / (std::sqrt(2.) * w));
  return {phi,
          (1. - phi * phi) / (std::sqrt(2.) * w),
          -phi * (1. - phi * phi) / (w * w)};
}

// Two interfaces of width epsilon at a distance a, enclosing phi > 0.
Profile1D filament(const double x, const double a, const double epsilon)
{
  const auto left  = tanh_profile(x + 0.5 * a, epsilon);
  const auto right = tanh_profile(0.5 * a - x, epsilon);
  return {left.phi + right.phi - 1.,
          left.dphi - right.dphi,
          left.ddphi + right.ddphi};
}

template <int dim>
CahnHilliard::InteractionIndicator
evaluate(const Profile1D &p, const double epsilon, const double sign = 1.)
{
  Tensor<1, dim> g;
  Tensor<2, dim> H;
  g[0]    = sign * p.dphi;
  H[0][0] = sign * p.ddphi;
  return CahnHilliard::evaluate_interaction_indicator<dim>(
    sign * p.phi, g, CahnHilliard::psi_hessian(sign * p.phi, g, H), epsilon);
}

template <int dim>
void check_planar_profiles()
{
  // Compressed, equilibrium and stretched profiles: Lambda vanishes and the
  // correction is kept, whatever the width.
  constexpr double epsilon = 0.3;
  for (const double width : {0.5, 1., 2.})
    for (const double x : {-1.2, -0.4, 0., 0.3, 0.9})
    {
      const auto p = tanh_profile(x * width * epsilon, width * epsilon);
      if (std::abs(p.phi) >= 0.95)
        continue;
      const auto r = evaluate<dim>(p, epsilon);
      AssertThrow(std::abs(r.lambda) < 1e-10 && r.positive == 0. &&
                    r.negative == 0.,
                  ExcMessage("A tanh profile must not trigger the cutoff."));
    }
  deallog << "dim " << dim << ": planar tanh profiles kept" << std::endl;
}

template <int dim>
void check_filaments()
{
  constexpr double epsilon = 0.3;
  for (const double a : {2., 3., 4.})
  {
    // At the centre, Lambda = -(3 + phi_c)/2 and the pinched phase is phi > 0.
    const auto   p     = filament(0., a * epsilon, epsilon);
    const auto   r     = evaluate<dim>(p, epsilon);
    const double exact = -0.5 * (3. + p.phi);
    AssertThrow(std::abs(r.lambda - exact) < 1e-10,
                ExcMessage("Wrong Lambda at the centre of a filament."));
    // The same filament made of the negative phase cuts the other side.
    const auto g = evaluate<dim>(p, epsilon, -1.);
    deallog << "dim " << dim << ", a = " << a << " eps: phi_c = " << p.phi
            << ", Lambda = " << r.lambda << ", weights (+,-) = (" << r.positive
            << ", " << r.negative << "), negative phase (" << g.positive << ", "
            << g.negative << ")" << std::endl;
  }
}

void check_fold()
{
  // Disk of radius R holding phi > 0, with an exact tanh profile in r: Lambda
  // vanishes, and kappa*eps = eps/R detects a radius below epsilon.
  constexpr double epsilon = 0.3;
  for (const double radius : {0.4, 10.})
  {
    const double R = radius * epsilon;
    const auto   p = tanh_profile(0., epsilon); // value at r = R
    Tensor<1, 2> g;
    Tensor<2, 2> H;
    g[0]         = -p.dphi;
    H[0][0]      = p.ddphi;
    H[1][1]      = -p.dphi / R;
    const auto r = CahnHilliard::evaluate_interaction_indicator<2>(
      p.phi, g, CahnHilliard::psi_hessian(p.phi, g, H), epsilon);
    deallog << "disk R = " << radius << " eps: Lambda = " << r.lambda
            << ", kappa eps = " << r.kappa_epsilon << ", weights (+,-) = ("
            << r.positive << ", " << r.negative << ")" << std::endl;
  }
}

int main()
{
  initlog();
  deallog << std::setprecision(6);
  check_planar_profiles<2>();
  check_planar_profiles<3>();
  check_filaments<2>();
  check_filaments<3>();
  check_fold();
}
