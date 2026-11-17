#include <chns_profile_coefficient.h>

#include "../tests.h"

using namespace dealii;

template <int dim>
void check()
{
  const double   eps = .3, base = .7, C = 1.2;
  Tensor<1, dim> w, dw;
  Tensor<2, dim> H, U, dH, dU;
  Tensor<3, dim> V, dV;
  w[0]      = 1. / (std::sqrt(2.) * eps);
  auto eval = [&]() {
    return CahnHilliard::LocalProfileCoefficient<dim>(w, H, U, V, eps, base, C);
  };
  AssertThrow(std::abs(eval().value - C * base) < 1e-14, ExcInternalError());
  U[0][1] = 2.;
  U[1][0] = -2.;
  AssertThrow(std::abs(eval().value - C * base) < 1e-14,
              ExcMessage("Rigid rotation changes strength"));
  U                  = {};
  U[0][0]            = 3.;
  const double delta = 1e-8 * base / (eps * eps), z = 3. / (1. + .05 * .05);
  AssertThrow(std::abs(eval().value -
                       C * (base + eps * eps *
                                     (std::hypot(z, delta) - delta))) < 1e-14,
              ExcMessage("Wrong normal strain"));
  const auto positive_strain = eval().value;
  U                          = -U;
  AssertThrow(std::abs(eval().value - positive_strain) < 1e-14,
              ExcMessage("Compression and extension must give the same rate"));
  w = -w;
  AssertThrow(std::abs(eval().value - positive_strain) < 1e-14,
              ExcMessage("Phase normal orientation must not change the rate"));
  const CahnHilliard::LocalProfileCoefficient<dim> disabled(
    w, H, U, V, eps, base, 0.);
  AssertThrow(disabled.value == 0. && disabled.gradient.norm() == 0.,
              ExcMessage("C=0 must disable profile amplitude"));
  for (unsigned i = 0; i < dim; ++i)
  {
    w[i]  = .7 + .2 * i;
    dw[i] = -.3 + .1 * i;
    for (unsigned j = 0; j < dim; ++j)
    {
      H[i][j]  = .4 + .1 * (i + j);
      dH[i][j] = -.1 + .03 * (i + j);
      U[i][j]  = .8 - .2 * i + .1 * j;
      dU[i][j] = .2 + .04 * i - .1 * j;
      for (unsigned k = 0; k < dim; ++k)
      {
        V[i][j][k]  = .3 + .2 * i + .1 * (j + k);
        dV[i][j][k] = -.2 + .1 * i + .03 * (j + k);
      }
    }
  }
  for (double scale : {1., 1e-6, 0.})
  {
    const auto                                       ws = scale * w;
    const CahnHilliard::LocalProfileCoefficient<dim> a(
      ws, H, U, V, eps, base, C);
    const auto   da = a.variation(dw, dH, dU, dV);
    const double h  = 1e-7;
    const CahnHilliard::LocalProfileCoefficient<dim> plus(
      ws + h * dw, H + h * dH, U + h * dU, V + h * dV, eps, base, C);
    const CahnHilliard::LocalProfileCoefficient<dim> minus(
      ws - h * dw, H - h * dH, U - h * dU, V - h * dV, eps, base, C);
    AssertThrow(std::abs((plus.value - minus.value) / (2 * h) - da.value) <
                  2e-7,
                ExcMessage("Coefficient derivative"));
    // At weak w, the smooth absolute value makes the third derivative large.
    // Keep h fixed and cancel the O(h^2) term with a fourth-order stencil.
    const CahnHilliard::LocalProfileCoefficient<dim> plus_half(ws + .5 * h * dw,
                                                               H + .5 * h * dH,
                                                               U + .5 * h * dU,
                                                               V + .5 * h * dV,
                                                               eps,
                                                               base,
                                                               C),
      minus_half(ws - .5 * h * dw,
                 H - .5 * h * dH,
                 U - .5 * h * dU,
                 V - .5 * h * dV,
                 eps,
                 base,
                 C);
    const auto centered = (plus.gradient - minus.gradient) / (2 * h);
    const auto fourth_order =
      (4. * (plus_half.gradient - minus_half.gradient) / h - centered) / 3.;
    AssertThrow((fourth_order - da.gradient).norm() < 2e-6,
                ExcMessage("Coefficient gradient derivative"));
    if (scale == 1e-6)
    {
      // Independent mixed derivative of the scalar K(t,x), computed with
      // 80 decimal digits; see reference_local_profile_derivative.py.
      const std::array<double, 3> reference =
        dim == 2 ?
          std::array<double, 3>{
            {-0.0278362140618073244785, -0.0340450451696826365726, 0.}} :
          std::array<double, 3>{{-0.137854485863888843968,
                                 -0.165690298551281987777,
                                 -0.193526111238675131586}};
      for (unsigned k = 0; k < dim; ++k)
        AssertThrow(std::abs(da.gradient[k] - reference[k]) < 2e-12,
                    ExcMessage("Independent weak-gradient reference"));
      const double error = (centered - da.gradient).norm();
      const double half_error =
        ((plus_half.gradient - minus_half.gradient) / h - da.gradient).norm();
      AssertThrow(error / half_error > 3.9 && error / half_error < 4.1,
                  ExcMessage(
                    "Expected second-order truncation at weak gradients"));
    }
    // Spatial derivative: w_x=H and (grad u)_x=V.
    for (unsigned k = 0; k < dim; ++k)
    {
      Tensor<1, dim> wx;
      Tensor<2, dim> Ux;
      for (unsigned i = 0; i < dim; ++i)
      {
        wx[i] = H[i][k];
        for (unsigned j = 0; j < dim; ++j)
          Ux[i][j] = V[i][j][k];
      }
      const CahnHilliard::LocalProfileCoefficient<dim> ap(
        ws + h * wx, H, U + h * Ux, V, eps, base, C),
        am(ws - h * wx, H, U - h * Ux, V, eps, base, C);
      AssertThrow(std::abs((ap.value - am.value) / (2 * h) - a.gradient[k]) <
                    2e-7,
                  ExcMessage("Spatial product rule"));
    }
  }
}
// Nonzero overlap and width feedback, including their spatial gradients.
// The geometry remains fixed with respect to Newton, as in assembly.
void check_geometry_coupling()
{
  constexpr double eps = .2, base = .7, C = 50., h = 1e-6;
  using Coefficient = CahnHilliard::LocalProfileCoefficient<2>;
  CHNSProfileOverlap::PenaltyValue geometry;
  geometry.value               = 3.;
  geometry.gradient            = {.4, -.3};
  geometry.feedback.multiplier = 2.3;
  geometry.feedback.gradient   = {-.2, .6};
  Tensor<1, 2> w, dw;
  Tensor<2, 2> H, U, dH, dU;
  Tensor<3, 2> V, dV;
  dw[0]   = .3;
  dw[1]   = -.2;
  H[0][0] = .1;
  H[0][1] = H[1][0] = .2;
  H[1][1]           = -.1;
  dH                = .3 * H;
  U[0][0]           = .3;
  U[1][1]           = -.5;
  dU                = .2 * U;
  V[0][0][0]        = .2;
  V[1][1][0]        = -.1;
  dV                = .4 * V;
  for (const double ratio : {0., .5, 1., 1.2, 2., 4.})
  {
    w[0] = ratio / (std::sqrt(2.) * eps);
    const Coefficient a(w, H, U, V, eps, base, C, geometry);
    const auto        da = a.variation(dw, dH, dU, dV);
    const Coefficient plus(
      w + h * dw, H + h * dH, U + h * dU, V + h * dV, eps, base, C, geometry),
      minus(
        w - h * dw, H - h * dH, U - h * dU, V - h * dV, eps, base, C, geometry);
    AssertThrow(std::abs((plus.value - minus.value) / (2 * h) - da.value) <
                  2e-7,
                ExcMessage("Overlap amplitude derivative"));
    const Coefficient ph(w + .5 * h * dw,
                         H + .5 * h * dH,
                         U + .5 * h * dU,
                         V + .5 * h * dV,
                         eps,
                         base,
                         C,
                         geometry),
      mh(w - .5 * h * dw,
         H - .5 * h * dH,
         U - .5 * h * dU,
         V - .5 * h * dV,
         eps,
         base,
         C,
         geometry);
    const auto coarse = (plus.gradient - minus.gradient) / (2 * h);
    const auto fine   = (ph.gradient - mh.gradient) / h;
    // At w=0 the local rate starts at O(|w|^4). Its spatial derivative
    // starts at O(|w|^3), giving O(h^2) in this centered Newton check.
    // Verify that truncation explicitly, then cancel it at the same h.
    if (ratio == 0.)
    {
      const double quotient =
        (coarse - da.gradient).norm() / (fine - da.gradient).norm();
      std::cout << "Zero-gradient mixed derivative: errors "
                << (coarse - da.gradient).norm() << " "
                << (fine - da.gradient).norm() << " ratio " << quotient
                << std::endl;
      AssertThrow(quotient > 3.9 && quotient < 4.1,
                  ExcMessage("Expected second-order zero-gradient truncation"));
    }
    AssertThrow(((4. * fine - coarse) / 3. - da.gradient).norm() < 2e-6,
                ExcMessage("Overlap mixed derivative"));
    for (unsigned k = 0; k < 2; ++k)
    {
      Tensor<1, 2> wx;
      Tensor<2, 2> Ux;
      for (unsigned i = 0; i < 2; ++i)
      {
        wx[i] = H[i][k];
        for (unsigned j = 0; j < 2; ++j)
          Ux[i][j] = V[i][j][k];
      }
      auto gp = geometry, gm = geometry;
      gp.value += h * geometry.gradient[k];
      gm.value -= h * geometry.gradient[k];
      gp.feedback.multiplier += h * geometry.feedback.gradient[k];
      gm.feedback.multiplier -= h * geometry.feedback.gradient[k];
      const Coefficient ap(w + h * wx, H, U + h * Ux, V, eps, base, C, gp),
        am(w - h * wx, H, U - h * Ux, V, eps, base, C, gm);
      AssertThrow(std::abs((ap.value - am.value) / (2 * h) - a.gradient[k]) <
                    2e-7,
                  ExcMessage("Overlap spatial derivative"));
    }
    AssertThrow(a.value > 0. && a.value <= a.unattenuated_value(),
                ExcInternalError());
    if (ratio <= 1.)
      AssertThrow(std::abs(a.value / a.unattenuated_value() - .25) < 1e-14,
                  ExcMessage("Overlap attenuation on an uncompressed profile"));
  }
}
int main()
{
  initlog();
  check<2>();
  check<3>();
  check_geometry_coupling();
  deallog << "Local profile coefficient and derivatives OK" << std::endl;
}
