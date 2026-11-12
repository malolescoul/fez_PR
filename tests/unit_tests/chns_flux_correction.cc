#include <chns_flux_correction.h>

#include "../tests.h"

template <int dim>
void check_projection()
{
  constexpr double epsilon = 0.3, mobility = 0.7;
  Tensor<1, dim>   normal, v;
  normal[0] = 1.;
  v[0]      = 0.8;
  v[1]      = 0.3;
  for (const double phi : {-0.999, -0.99, 0., 0.99, 0.999})
  {
    const auto g = (1. - phi * phi) / (std::sqrt(2.) * epsilon) * normal;
    const CahnHilliard::FluxProjection<dim> projection(
      phi, g, {}, v, {}, mobility, {}, epsilon, false);
    const auto flux = mobility * v + projection.flux;
    AssertThrow(std::abs(flux[0]) < 1e-14 &&
                  std::abs(flux[1] - mobility * v[1]) < 1e-14,
                ExcMessage("Only the normal chemical flux must be removed."));
  }
  const double lower =
    CahnHilliard::profile_correction_tail / (4. * std::sqrt(2.) * epsilon);
  // At the midpoint of the gradient transition chi=1/2: the normal
  // flux is attenuated by chi^2, not chi.
  const CahnHilliard::FluxProjection<dim> partial(
    0., 1.5 * lower * normal, {}, v, {}, mobility, {}, epsilon, false);
  AssertThrow(std::abs(partial.flux[0] + 0.25 * mobility * v[0]) < 1e-14,
              ExcMessage("The weighted normal must appear twice."));
  for (const double phi : {0., 1., -1., 1.01, -1.01})
    for (const double magnitude : {0., 0.5 * lower, 1.})
      if (magnitude < lower || std::abs(phi) >= 1.)
      {
        const CahnHilliard::FluxProjection<dim> inactive(
          phi, magnitude * normal, {}, v, {}, mobility, {}, epsilon, true);
        const auto variation =
          inactive.variation({1., normal, {}}, v, {}, 1., normal);
        AssertThrow(inactive.flux.norm() == 0. && inactive.divergence == 0. &&
                      variation.flux.norm() == 0. && variation.divergence == 0.,
                    ExcMessage(
                      "Inactive projection must remain finite and zero."));
      }
}

template <int dim>
void check_derivatives()
{
  constexpr double epsilon = 0.3, M = 0.7;
  Tensor<1, dim>   direction, v, grad_M;
  Tensor<2, dim>   H, T;
  for (unsigned int i = 0; i < dim; ++i)
  {
    direction[i] = 0.2 + 0.1 * i;
    v[i]         = 0.4 - 0.1 * i;
    grad_M[i]    = 0.03 * (i + 1);
    for (unsigned int j = 0; j < dim; ++j)
    {
      H[i][j] = 0.01 * (i + j + 1);
      T[i][j] = 0.02 * (i + j + 2);
    }
  }
  direction /= direction.norm();
  const double lower =
    CahnHilliard::profile_correction_tail / (4. * std::sqrt(2.) * epsilon);
  for (const double phi : {0., -0.4, -0.9996, 0.9996, 1.01})
    for (const double magnitude : {0., 1.5 * lower, 0.4})
    {
      const auto                              g = magnitude * direction;
      const CahnHilliard::FluxProjection<dim> projection(
        phi, g, H, v, T, M, grad_M, epsilon, true);
      // Evaluate the flux at nearby spatial points of quadratic r and mu,
      // and affine M; no divergence expression is used in this check.
      double           numerical_divergence = 0.;
      constexpr double h                    = 1e-8;
      for (unsigned int d = 0; d < dim; ++d)
      {
        Tensor<1, dim> dx;
        dx[d] = h;
        const CahnHilliard::FluxProjection<dim> plus(phi + g * dx +
                                                       0.5 * dx * (H * dx),
                                                     g + H * dx,
                                                     H,
                                                     v + T * dx,
                                                     T,
                                                     M + grad_M * dx,
                                                     grad_M,
                                                     epsilon,
                                                     false);
        const CahnHilliard::FluxProjection<dim> minus(phi - g * dx +
                                                        0.5 * dx * (H * dx),
                                                      g - H * dx,
                                                      H,
                                                      v - T * dx,
                                                      T,
                                                      M - grad_M * dx,
                                                      grad_M,
                                                      epsilon,
                                                      false);
        numerical_divergence += (plus.flux[d] - minus.flux[d]) / (2. * h);
      }
      AssertThrow(std::abs(numerical_divergence - projection.divergence) <
                    2e-6 * std::max(1., std::abs(projection.divergence)),
                  ExcMessage(
                    "Projection divergence differs from spatial differences."));
      // Perturb each independent argument, including both Hessians and grad(M).
      for (unsigned int k = 0; k < 2 + 3 * dim + 2 * dim * dim; ++k)
      {
        CahnHilliard::ProfileVariation<dim> dr{};
        Tensor<1, dim>                      dv, dgrad_M;
        Tensor<2, dim>                      dT;
        double                              dM = 0.;
        if (k == 0)
          dr.value = 1.;
        else if (k == 1)
          dM = 1.;
        else if (k < 2 + dim)
          dr.gradient[k - 2] = 1.;
        else if (k < 2 + 2 * dim)
          dv[k - 2 - dim] = 1.;
        else if (k < 2 + 3 * dim)
          dgrad_M[k - 2 - 2 * dim] = 1.;
        else
        {
          const unsigned int n      = k - 2 - 3 * dim;
          auto              &tensor = n < dim * dim ? dr.hessian : dT;
          const unsigned int index  = n % (dim * dim);
          tensor[index / dim][index % dim] += 0.5;
          tensor[index % dim][index / dim] += 0.5;
        }
        const auto analytic = projection.variation(dr, dv, dT, dM, dgrad_M);
        double     error    = std::numeric_limits<double>::max();
        for (const double step : {1e-6, 1e-7, 1e-8})
        {
          const CahnHilliard::FluxProjection<dim> plus(phi + step * dr.value,
                                                       g + step * dr.gradient,
                                                       H + step * dr.hessian,
                                                       v + step * dv,
                                                       T + step * dT,
                                                       M + step * dM,
                                                       grad_M + step * dgrad_M,
                                                       epsilon,
                                                       true);
          const CahnHilliard::FluxProjection<dim> minus(phi - step * dr.value,
                                                        g - step * dr.gradient,
                                                        H - step * dr.hessian,
                                                        v - step * dv,
                                                        T - step * dT,
                                                        M - step * dM,
                                                        grad_M - step * dgrad_M,
                                                        epsilon,
                                                        true);
          error = std::min(
            error,
            std::max(
              ((plus.flux - minus.flux) / (2. * step) - analytic.flux).norm() /
                std::max(1., analytic.flux.norm()),
              std::abs((plus.divergence - minus.divergence) / (2. * step) -
                       analytic.divergence) /
                std::max(1., std::abs(analytic.divergence))));
        }
        AssertThrow(error < 3e-6,
                    ExcMessage("Incorrect projection linearization."));
      }
    }
}

int main()
{
  initlog();
  check_projection<2>();
  check_projection<3>();
  check_derivatives<2>();
  check_derivatives<3>();
  deallog << "Flux projection: normal, tangent, transitions, divergence and "
             "derivatives OK"
          << std::endl;
}
