#ifndef CHNS_SURFACE_TENSION_H
#define CHNS_SURFACE_TENSION_H

#include <deal.II/base/tensor.h>

/** Linear-weight diffuse surface stress. No floor is added to |grad(phi)|.
 * At zero gradient the stress is zero; Newton uses the zero generalized
 * derivative. The elementwise divergence is for residual stabilization only:
 * visualization must retain interelement stress jumps through weak recovery.
 */
namespace CHNSSurfaceTension
{
  using namespace dealii;

  template <int dim>
  Tensor<2, dim> stress(const Tensor<1, dim> &g, const double sigma)
  {
    Tensor<2, dim> T;
    const double   r = g.norm();
    if (r == 0.)
      return T;
    const auto n = g / r;
    T            = -outer_product(n, n);
    for (unsigned int d = 0; d < dim; ++d)
      T[d][d] += 1.;
    return (0.5 * sigma * r) * T;
  }

  template <int dim>
  Tensor<2, dim> stress_variation(const Tensor<1, dim> &g,
                                  const Tensor<1, dim> &dg,
                                  const double          sigma)
  {
    Tensor<2, dim> dT;
    const double   r = g.norm();
    if (r == 0.)
      return dT;
    const auto   n  = g / r;
    const double dr = n * dg;
    dT = dr * outer_product(n, n) - outer_product(dg, n) - outer_product(n, dg);
    for (unsigned int d = 0; d < dim; ++d)
      dT[d][d] += dr;
    return (0.5 * sigma) * dT;
  }

  // div(T) = -sigma/2 n (trace(H) - n.H.n), for smooth phi.
  template <int dim>
  Tensor<1, dim>
  force(const Tensor<1, dim> &g, const Tensor<2, dim> &H, const double sigma)
  {
    const double r = g.norm();
    if (r == 0.)
      return {};
    const auto n = g / r;
    return (-0.5 * sigma * (trace(H) - n * (H * n))) * n;
  }

  template <int dim>
  Tensor<1, dim> force_variation(const Tensor<1, dim> &g,
                                 const Tensor<2, dim> &H,
                                 const Tensor<1, dim> &dg,
                                 const Tensor<2, dim> &dH,
                                 const double          sigma)
  {
    const double r = g.norm();
    if (r == 0.)
      return {};
    const auto n  = g / r;
    const auto dn = (dg - (n * dg) * n) / r;
    return -0.5 * sigma *
           ((trace(H) - n * (H * n)) * dn +
            (trace(dH) - dn * (H * n) - n * (H * dn) - n * (dH * n)) * n);
  }
} // namespace CHNSSurfaceTension
#endif
