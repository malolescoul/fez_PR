#ifndef CHNS_PROFILE_MAPPING_H
#define CHNS_PROFILE_MAPPING_H

#include <chns_profile_overlap.h>

namespace CHNSProfileGeometry
{
  /** Immutable physical cell map. Triangle P1/P2 and tensor Q1/Q2 positions;
   * the transported phase may still be linear. No Newton state is retained. */
  struct CellMap
  {
    using Point                  = CHNSProfileOverlap::Point;
    bool                 simplex = false;
    unsigned             degree  = 1;
    std::array<Point, 9> nodes{};

    unsigned size() const
    {
      return simplex ? (degree == 1 ? 3 : 6) : (degree + 1) * (degree + 1);
    }
    Point support_point(unsigned k) const
    {
      if (simplex)
      {
        static constexpr std::array<Point, 6> p{
          {{0., 0.}, {1., 0.}, {0., 1.}, {.5, 0.}, {0., .5}, {.5, .5}}};
        return p[k];
      }
      return {double(k % (degree + 1)) / degree,
              double(k / (degree + 1)) / degree};
    }
    Point evaluate(Point p, std::array<Point, 2> *columns = nullptr) const
    {
      return CHNSProfileOverlap::add(nodes[0], evaluate_offset(p, columns));
    }
    // Form both the map and its derivatives from differences of nodal
    // positions. Absolute coordinates can be much larger than the cell.
    Point evaluate_offset(Point                 p,
                          std::array<Point, 2> *columns = nullptr) const
    {
      using namespace CHNSProfileOverlap;
      std::array<double, 9> w{}, dx{}, dy{};
      if (simplex)
      {
        const std::array<double, 3> l{{1. - p.x - p.y, p.x, p.y}};
        const std::array<Point, 3>  dl{{{-1., -1.}, {1., 0.}, {0., 1.}}};
        for (unsigned i = 0; i < 3; ++i)
        {
          w[i]  = degree == 1 ? l[i] : l[i] * (2. * l[i] - 1.);
          dx[i] = dl[i].x * (degree == 1 ? 1. : 4. * l[i] - 1.);
          dy[i] = dl[i].y * (degree == 1 ? 1. : 4. * l[i] - 1.);
        }
        if (degree == 2)
          for (unsigned k = 0; k < 3; ++k)
          {
            const unsigned a = k == 2 ? 1 : 0, b = k == 0 ? 1 : 2;
            w[k + 3]  = 4. * l[a] * l[b];
            dx[k + 3] = 4. * (dl[a].x * l[b] + l[a] * dl[b].x);
            dy[k + 3] = 4. * (dl[a].y * l[b] + l[a] * dl[b].y);
          }
      }
      else
      {
        const auto basis = [&](double x) {
          return degree == 1 ? std::array<double, 3>{{1. - x, x, 0.}} :
                               std::array<double, 3>{{(1. - x) * (1. - 2. * x),
                                                      4. * x * (1. - x),
                                                      x * (2. * x - 1.)}};
        };
        const auto derivative = [&](double x) {
          return degree == 1 ? std::array<double, 3>{{-1., 1., 0.}} :
                               std::array<double, 3>{
                                 {4. * x - 3., 4. - 8. * x, 4. * x - 1.}};
        };
        const auto x = basis(p.x), y = basis(p.y), gx = derivative(p.x),
                   gy = derivative(p.y);
        for (unsigned j = 0; j <= degree; ++j)
          for (unsigned i = 0; i <= degree; ++i)
          {
            const unsigned k = i + (degree + 1) * j;
            w[k]             = x[i] * y[j];
            dx[k]            = gx[i] * y[j];
            dy[k]            = x[i] * gy[j];
          }
      }
      Point value;
      if (columns)
        *columns = {};
      for (unsigned k = 0; k < size(); ++k)
      {
        const auto relative = subtract(nodes[k], nodes[0]);
        value               = add(value, scale(relative, w[k]));
        if (columns)
        {
          (*columns)[0] = add((*columns)[0], scale(relative, dx[k]));
          (*columns)[1] = add((*columns)[1], scale(relative, dy[k]));
        }
      }
      return value;
    }
    // Bernstein control points bound even curved cells, unlike vertex boxes.
    std::pair<Point, Point> bounds() const
    {
      using namespace CHNSProfileOverlap;
      auto control = nodes;
      if (degree == 2)
      {
        if (simplex)
          for (unsigned k = 0; k < 3; ++k)
          {
            const unsigned a = k == 2 ? 1 : 0, b = k == 0 ? 1 : 2;
            control[k + 3] = subtract(scale(nodes[k + 3], 2.),
                                      scale(add(nodes[a], nodes[b]), .5));
          }
        else
        {
          for (unsigned j = 0; j < 3; ++j)
            control[1 + 3 * j] =
              subtract(scale(control[1 + 3 * j], 2.),
                       scale(add(control[3 * j], control[2 + 3 * j]), .5));
          for (unsigned i = 0; i < 3; ++i)
            control[i + 3] =
              subtract(scale(control[i + 3], 2.),
                       scale(add(control[i], control[i + 6]), .5));
        }
      }
      Point lo{INFINITY, INFINITY}, hi{-INFINITY, -INFINITY};
      for (unsigned k = 0; k < size(); ++k)
        for (unsigned d = 0; d < 2; ++d)
        {
          lo[d] = std::min(lo[d], control[k][d]);
          hi[d] = std::max(hi[d], control[k][d]);
        }
      return {lo, hi};
    }
    bool inverse(Point physical, Point &unit) const
    {
      using namespace CHNSProfileOverlap;
      unit             = simplex ? Point{1. / 3., 1. / 3.} : Point{.5, .5};
      const auto   box = bounds();
      const double diameter2 =
        dot(subtract(box.second, box.first), subtract(box.second, box.first));
      const double tolerance =
        1e-13 * std::sqrt(diameter2) +
        8. * std::numeric_limits<double>::epsilon() *
          std::max(
            {std::abs(physical.x), std::abs(physical.y), std::sqrt(diameter2)});
      for (unsigned it = 0; it < 20; ++it)
      {
        std::array<Point, 2> J;
        const auto           residual =
          subtract(evaluate_offset(unit, &J), subtract(physical, nodes[0]));
        const double det = cross(J[0], J[1]);
        if (std::abs(det) < 1e-14 * diameter2)
          return false;
        if (dot(residual, residual) <= tolerance * tolerance)
          return true;
        const Point delta{cross(residual, J[1]) / det,
                          cross(J[0], residual) / det};
        unit = subtract(unit, delta);
        if (!std::isfinite(unit.x + unit.y))
          return false;
      }
      return false;
    }
    bool contains(Point physical) const
    {
      Point p;
      if (!inverse(physical, p))
        return false;
      using namespace CHNSProfileOverlap;
      std::array<Point, 2> J;
      evaluate(p, &J);
      // Include the reference-space uncertainty of a physical coordinate.
      // Inverse convergence and containment must agree at shared edges.
      const double roundoff =
        8. * std::numeric_limits<double>::epsilon() *
        std::max(std::abs(physical.x), std::abs(physical.y));
      const double tol = 1e-10 + roundoff *
                                   (std::abs(J[0].x) + std::abs(J[0].y) +
                                    std::abs(J[1].x) + std::abs(J[1].y)) /
                                   std::abs(cross(J[0], J[1]));
      return p.x >= -tol && p.y >= -tol &&
             (simplex ? p.x + p.y <= 1. + tol :
                        p.x <= 1. + tol && p.y <= 1. + tol);
    }
  };
} // namespace CHNSProfileGeometry
#endif
