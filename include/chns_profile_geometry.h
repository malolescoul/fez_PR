#ifndef CHNS_PROFILE_GEOMETRY_H
#define CHNS_PROFILE_GEOMETRY_H
#include <chns_profile_overlap.h>
#include <chns_profile_width.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace CHNSProfileOverlap
{
  struct PenaltyValue
  {
    double                  value = 0.;
    Point                   gradient;
    CHNSProfileWidth::Value feedback;
  };

  struct ReflectingBox
  {
    Point lower, upper;
    Point axis{1., 0.}; // First axis of an orthonormal physical frame.
    std::array<bool, 2> reflect_lower{{false, false}},
      reflect_upper{{false, false}};
    Point local(Point p) const
    {
      return {axis.x * p.x + axis.y * p.y, -axis.y * p.x + axis.x * p.y};
    }
    Point physical(Point p) const
    {
      return {axis.x * p.x - axis.y * p.y, axis.y * p.x + axis.x * p.y};
    }
  };

  inline std::vector<MidlineSource>
  extend_sources(std::vector<MidlineSource> input,
                 const ReflectingBox       &box,
                 const double               radius)
  {
    if (!box.reflect_lower[0] && !box.reflect_lower[1] &&
        !box.reflect_upper[0] && !box.reflect_upper[1])
      return input;
    const bool rotated = box.axis.x != 1. || box.axis.y != 0.;
    if (rotated)
      for (auto &s : input)
      {
        s.a = box.local(s.a);
        s.b = box.local(s.b);
      }
    for (unsigned axis = 0; axis < 2; ++axis)
    {
      if (!box.reflect_lower[axis] && !box.reflect_upper[axis])
        continue;
      const double lo = box.lower[axis], hi = box.upper[axis], length = hi - lo;
      if (!(length > 0.))
        throw std::invalid_argument("Invalid reflecting box");
      std::vector<MidlineSource> next;
      for (const auto &s : input)
      {
        const auto append = [&](double sign, double shift) {
          auto a = s.a, b = s.b;
          if (axis == 0)
          {
            a.x = sign * a.x + shift;
            b.x = sign * b.x + shift;
          }
          else
          {
            a.y = sign * a.y + shift;
            b.y = sign * b.y + shift;
          }
          if (std::max(a[axis], b[axis]) < lo - radius ||
              std::min(a[axis], b[axis]) > hi + radius)
            return;
          // A segment lying on a reflecting plane must not be counted twice.
          if (sign < 0. &&
              std::abs(a[axis] - s.a[axis]) + std::abs(b[axis] - s.b[axis]) <
                1e-12 * length)
            return;
          auto reflected = s;
          reflected.a    = a;
          reflected.b    = b;
          next.push_back(reflected);
        };
        if (box.reflect_lower[axis] && box.reflect_upper[axis])
        {
          const int count =
            static_cast<int>(std::ceil(radius / (2. * length))) + 2;
          for (int k = -count; k <= count; ++k)
          {
            append(1., 2. * k * length);
            append(-1., 2. * lo + 2. * k * length);
          }
        }
        else
        {
          append(1., 0.);
          append(-1., 2. * (box.reflect_lower[axis] ? lo : hi));
        }
      }
      input = std::move(next);
    }
    if (rotated)
      for (auto &s : input)
      {
        s.a = box.physical(s.a);
        s.b = box.physical(s.b);
      }
    return input;
  }

  /** Smooth convolution of paired-interface midline segments. No local
   * profile Hessian is used. The line integral is analytic, so subdividing
   * a straight segment does not produce ripples in the strength field.
   */
  class SmoothPenalty
  {
  public:
    SmoothPenalty(std::vector<MidlineSource> sources,
                  double                     epsilon,
                  double               width_in_epsilon = 1. / std::sqrt(6.),
                  const ReflectingBox &box              = {})
      : eps(epsilon)
      , ell(epsilon * width_in_epsilon)
      , support(3. * std::sqrt(6.) * ell)
      , sources(std::move(sources))
    {
      this->sources = extend_sources(std::move(this->sources), box, support);
      for (std::size_t i = 0; i < this->sources.size(); ++i)
      {
        const auto &s  = this->sources[i];
        const auto  lo = bin(
          {std::min(s.a.x, s.b.x) - support, std::min(s.a.y, s.b.y) - support});
        const auto hi = bin(
          {std::max(s.a.x, s.b.x) + support, std::max(s.a.y, s.b.y) + support});
        for (int x = lo.first; x <= hi.first; ++x)
          for (int y = lo.second; y <= hi.second; ++y)
            bins[{x, y}].push_back(i);
      }
    }

    std::size_t  size() const { return sources.size(); }
    PenaltyValue evaluate(Point p) const
    {
      if (sources.empty() || bins.find(bin(p)) == bins.end())
        return {};
      Key key;
      std::memcpy(&key.x, &p.x, sizeof(double));
      std::memcpy(&key.y, &p.y, sizeof(double));
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto                  found = cache.find(key);
        if (found != cache.end())
          return found->second;
      }
      const auto result = evaluate_uncached(p);
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(key, result);
      }
      return result;
    }

  private:
    // Geometry is immutable for the entire Newton solve. Reusing its
    // value at a quadrature point changes neither the law nor its derivatives.
    struct Key
    {
      std::uint64_t x, y;
      bool operator==(const Key &k) const { return x == k.x && y == k.y; }
    };
    struct KeyHash
    {
      std::size_t operator()(const Key &k) const
      {
        return k.x ^ (k.y + 0x9e3779b97f4a7c15ULL + (k.x << 6) + (k.x >> 2));
      }
    };
    mutable std::mutex                                     cache_mutex;
    mutable std::unordered_map<Key, PenaltyValue, KeyHash> cache;
    PenaltyValue evaluate_uncached(Point p) const
    {
      PenaltyValue result;
      const auto   found = bins.find(bin(p));
      if (found == bins.end())
        return result;
      for (const auto i : found->second)
      {
        const auto  &s      = sources[i];
        const auto   edge   = subtract(s.b, s.a);
        const double length = std::sqrt(dot(edge, edge));
        if (length <= 1e-12 * eps)
          continue;
        const auto   tangent = scale(edge, 1. / length);
        const Point  normal{-tangent.y, tangent.x};
        const auto   d = subtract(p, scale(add(s.a, s.b), .5));
        const double a = dot(d, tangent), b = dot(d, normal);
        const double outside = std::max(0., std::abs(a) - .5 * length);
        const double d2      = b * b + outside * outside;
        if (d2 >= support * support)
          continue;
        const double right = (a + .5 * length) / ell,
                     left  = (a - .5 * length) / ell;
        const double e     = std::exp(-b * b / (ell * ell));
        const double erf_diff =
          left > 0.  ? std::erfc(left) - std::erfc(right) :
          right < 0. ? std::erfc(-right) - std::erfc(-left) :
                       std::erf(right) - std::erf(left);
        double value = .5 * e * erf_diff;
        Point  gradient =
          add(scale(tangent,
                    e * (std::exp(-right * right) - std::exp(-left * left)) /
                      (std::sqrt(std::acos(-1.)) * ell)),
              scale(normal, -2. * b * value / (ell * ell)));
        if (s.confine_to_film)
        {
          // Two facing-side windows: one epsilon on either side of each
          // zero-equivalent front. Their product is even and C2 at the median.
          const auto window = [](double z) {
            if (z <= 0.)
              return std::pair<double, double>{0., 0.};
            if (z >= 1.)
              return std::pair<double, double>{1., 0.};
            return std::pair<double, double>{z * z * z *
                                               (10. + z * (-15. + 6. * z)),
                                             30. * z * z * (z - 1.) * (z - 1.)};
          };
          const auto left_window  = window((s.half_gap + b + eps) / (2. * eps));
          const auto right_window = window((s.half_gap - b + eps) / (2. * eps));
          const double inside     = left_window.first * right_window.first;
          const double derivative = (left_window.second * right_window.first -
                                     left_window.first * right_window.second) /
                                    (2. * eps);
          gradient =
            add(scale(gradient, inside), scale(normal, value * derivative));
          value *= inside;
        }
        // C1 compact support; discarded Gaussian tails are below exp(-37.5).
        if (d2 > 37.5 * ell * ell)
        {
          const double z    = (d2 - 37.5 * ell * ell) / (16.5 * ell * ell);
          const double fade = 1. - z * z * z * (10. + z * (-15. + 6. * z));
          const double df =
            -30. * z * z * (z - 1.) * (z - 1.) / (16.5 * ell * ell);
          const auto gd2 = add(scale(normal, 2. * b),
                               scale(tangent, 2. * std::copysign(outside, a)));
          gradient       = add(scale(gradient, fade), scale(gd2, value * df));
          value *= fade;
        }
        result.value += s.weight * value / (.05 * .05);
        result.gradient =
          add(result.gradient, scale(gradient, s.weight / (.05 * .05)));
      }
      return result;
    }

  private:
    std::pair<int, int> bin(Point p) const
    {
      return {int(std::floor(p.x / support)), int(std::floor(p.y / support))};
    }
    double                                                  eps, ell, support;
    std::vector<MidlineSource>                              sources;
    std::map<std::pair<int, int>, std::vector<std::size_t>> bins;
  };

} // namespace CHNSProfileOverlap

namespace CHNSProfileGeometry
{
  /** Per-solver geometric fields. Shared read-only by assembly and output. */
  struct Fields
  {
    std::shared_ptr<const CHNSProfileOverlap::SmoothPenalty> overlap;
    std::shared_ptr<const CHNSProfileWidth::Geometry>        width;
    // Empty on a fixed mesh. ALE fields are frozen on the attempt's reference
    // configuration; their covectors follow the current physical mapping.
    std::unordered_map<std::string, CellMap> reference_cells;

    template <typename PhysicalPoint>
    CHNSProfileOverlap::PenaltyValue evaluate(const PhysicalPoint &point) const
    {
      const CHNSProfileOverlap::Point p{point[0], point[1]};
      auto                            result =
        overlap ? overlap->evaluate(p) : CHNSProfileOverlap::PenaltyValue{};
      if (width)
        result.feedback = width->evaluate(p);
      return result;
    }

    template <typename InverseJacobian>
    CHNSProfileOverlap::PenaltyValue
    evaluate_on_cell(const std::string              &id,
                     const CHNSProfileOverlap::Point unit,
                     const InverseJacobian          &inverse_current) const
    {
      const auto                              &map = reference_cells.at(id);
      std::array<CHNSProfileOverlap::Point, 2> J;
      auto       result    = evaluate(map.evaluate(unit, &J));
      const auto transport = [&](CHNSProfileOverlap::Point gradient) {
        const std::array<double, 2> reference{
          {CHNSProfileOverlap::dot(J[0], gradient),
           CHNSProfileOverlap::dot(J[1], gradient)}};
        CHNSProfileOverlap::Point current;
        for (unsigned i = 0; i < 2; ++i)
          for (unsigned j = 0; j < 2; ++j)
            current[i] += inverse_current[j][i] * reference[j];
        return current;
      };
      result.gradient          = transport(result.gradient);
      result.feedback.gradient = transport(result.feedback.gradient);
      return result;
    }
  };
} // namespace CHNSProfileGeometry
#endif
