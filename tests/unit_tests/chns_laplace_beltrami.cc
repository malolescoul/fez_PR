#include <chns_surface_tension.h>

#include "../tests.h"

using namespace dealii;
template <int dim>
void check()
{
  using namespace CHNSSurfaceTension;
  Tensor<1, dim> g, dg;
  Tensor<2, dim> H, dH;
  for (unsigned i = 0; i < dim; ++i)
  {
    g[i]  = 0.6 + 0.3 * i;
    dg[i] = 0.2 - 0.3 * i;
    for (unsigned j = 0; j < dim; ++j)
    {
      H[i][j]  = 0.2 + 0.1 * (i + j);
      dH[i][j] = 0.4 - 0.2 * (i + j);
    }
  }
  const double sigma = 0.072, h = 1e-6;
  AssertThrow((stress(g, sigma) * g).norm() < 1e-14, ExcInternalError());
  AssertThrow(stress(Tensor<1, dim>(), sigma).norm() == 0., ExcInternalError());
  AssertThrow(force(Tensor<1, dim>(), H, sigma).norm() == 0.,
              ExcInternalError());
  AssertThrow((stress_variation(g, dg, sigma) -
               (stress(g + h * dg, sigma) - stress(g - h * dg, sigma)) /
                 (2 * h))
                  .norm() < 1e-10,
              ExcInternalError());
  AssertThrow((force_variation(g, H, dg, dH, sigma) -
               (force(g + h * dg, H + h * dH, sigma) -
                force(g - h * dg, H - h * dH, sigma)) /
                 (2 * h))
                  .norm() < 1e-10,
              ExcInternalError());
  // Hessian along a planar normal must produce no capillary force.
  AssertThrow(force(g, outer_product(g, g), sigma).norm() < 1e-14,
              ExcInternalError());
  // Sphere: g=e_r, H=(I-e_r e_r)/R, so f=-sigma*(dim-1)/(2R)*e_r.
  g    = {};
  H    = {};
  g[0] = 1.;
  for (unsigned i = 1; i < dim; ++i)
    H[i][i] = 0.5;
  AssertThrow((force(g, H, sigma) + sigma * (dim - 1) / 4. * g).norm() < 1e-14,
              ExcInternalError());
  // Integral of exact linear weight across tanh is independent of width.
  for (double width : {0.5, 1., 2.})
  {
    double       integral = 0.;
    const double ds       = 40. * width / 20000.;
    for (unsigned k = 0; k < 20000; ++k)
    {
      const double s = -20. * width + (k + 0.5) * ds;
      const double c = std::cosh(s / width);
      integral += 0.5 / (width * c * c) * ds;
    }
    AssertThrow(std::abs(integral - 1.) < 1e-10, ExcInternalError());
  }
  deallog << dim << "D linear-weight stress and derivatives OK" << std::endl;
}
int main()
{
  initlog();
  check<2>();
  check<3>();
}
