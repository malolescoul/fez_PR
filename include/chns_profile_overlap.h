#ifndef CHNS_PROFILE_OVERLAP_H
#define CHNS_PROFILE_OVERLAP_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace CHNSProfileOverlap
{
  struct Point
  {
    double x = 0., y = 0.;
    double operator[](std::size_t i) const { return i == 0 ? x : y; }
  };
  inline Point add(Point a, Point b)
  {
    return {a[0] + b[0], a[1] + b[1]};
  }
  inline Point scale(Point a, double s)
  {
    return {s * a[0], s * a[1]};
  }
  inline Point subtract(Point a, Point b)
  {
    return add(a, scale(b, -1.));
  }
  inline double dot(Point a, Point b)
  {
    return a[0] * b[0] + a[1] * b[1];
  }
  inline double cross(Point a, Point b)
  {
    return a[0] * b[1] - a[1] * b[0];
  }

  struct Segment
  {
    Point a, b, normal; // normal points towards increasing phi
  };

  struct MidlineSource
  {
    Point  a, b;
    double weight          = 1.;
    double half_gap        = 0.;
    bool   confine_to_film = false;
  };

  // Geometric pairing quality; ambiguous contacts do not contribute.
  enum class Status
  {
    unavailable = 0,
    no_partner  = 1,
    measured    = 2,
    ambiguous   = 3
  };

  struct Result
  {
    double gap = 0., overlap = 0.;
    Point  partner_direction;
    double facing_coordinate = 0.;
    bool   valid             = false;
    Status status            = Status::unavailable;
  };

  /** Pair nonadjacent, facing contour segments to locate interacting films.
   * Pairing uses the validated exterior +/-0.99 contours; ambiguous contacts
   * do not contribute to the attenuation field.
   */
  class Geometry
  {
  public:
    Geometry(std::vector<Segment> contour,
             double               epsilon,
             double               search_radius_in_epsilon,
             double               allowed_phase = 0.)
      : eps(epsilon)
      , radius(epsilon * search_radius_in_epsilon)
      , pairing_phase(allowed_phase)
    {
      if (!(eps > 0.) || !(radius > 0.) || !std::isfinite(radius))
        throw std::invalid_argument(
          "Interface proximity requires positive finite scales");
      build_connectivity(std::move(contour));
      for (std::size_t i = 0; i < segments.size(); ++i)
      {
        const auto &s  = segments[i];
        const auto  lo = bin({std::min(s.a.x, s.b.x), std::min(s.a.y, s.b.y)});
        const auto  hi = bin({std::max(s.a.x, s.b.x), std::max(s.a.y, s.b.y)});
        for (int x = lo.first; x <= hi.first; ++x)
          for (int y = lo.second; y <= hi.second; ++y)
            bins[{x, y}].push_back(i);
      }
      for (std::size_t i = 0; i < segments.size(); ++i)
        gaps.push_back(measure(i));
    }

    std::vector<MidlineSource>
    midline_sources(const double level,
                    const bool   weight_overlap  = false,
                    const bool   confine_to_film = false) const
    {
      std::vector<MidlineSource> result;
      const double tail_distance = std::sqrt(2.) * eps * std::atanh(.9999);
      for (std::size_t i = 0; i < segments.size(); ++i)
      {
        const auto &g = gaps[i];
        if (!g.valid)
          continue;
        const double inside =
          dot(g.partner_direction, segments[i].normal) > 0. ? 1. : -1.;
        // Use the exterior level only. It survives the loss of the zero level.
        if (inside * level >= 0.)
          continue;
        const double gap =
          g.gap + 2. * inside * std::sqrt(2.) * eps * std::atanh(level);
        const double z = (gap - 2. * tail_distance) / (2. * eps);
        if (z >= 1.)
          continue;
        const double fade =
          z <= 0. ? 1. : 1. - z * z * z * (10. + z * (-15. + 6. * z));
        const auto shift = scale(g.partner_direction, .5 * g.gap);
        // Each pair is sampled from both interfaces, hence the factor 1/2.
        // The detector is on an exterior contour. Use the zero-equivalent
        // gap above for the same overlap indicator as the original mask.
        // Continue smoothly for negative equivalent gaps after zero is lost;
        // eta then lies between one and two, with no new clipping threshold.
        const double eta = weight_overlap ?
                             1. - std::tanh(gap / (2. * std::sqrt(2.) * eps)) :
                             1.;
        result.push_back({add(segments[i].a, shift),
                          add(segments[i].b, shift),
                          .5 * fade * eta,
                          .5 * gap,
                          confine_to_film});
      }
      return result;
    }

  private:
    using Bin     = std::pair<int, int>;
    using NodeBin = std::pair<long long, long long>;

    void build_connectivity(std::vector<Segment> contour)
    {
      // Merge roundoff-level discrepancies across cell and MPI boundaries.
      // AMR extraction gaps larger than this tolerance are NOT bridged.
      double scale_coordinate = eps;
      for (const auto &s : contour)
        for (const auto &p : {s.a, s.b})
          scale_coordinate =
            std::max({scale_coordinate, std::abs(p.x), std::abs(p.y)});
      const double tolerance =
        std::max(1e-8 * eps,
                 64. * std::numeric_limits<double>::epsilon() *
                   scale_coordinate);
      std::vector<Point>                          nodes;
      std::map<NodeBin, std::vector<std::size_t>> node_bins;
      const auto node_id = [&](const Point &p) {
        const NodeBin b{static_cast<long long>(std::floor(p.x / tolerance)),
                        static_cast<long long>(std::floor(p.y / tolerance))};
        for (long long x = b.first - 1; x <= b.first + 1; ++x)
          for (long long y = b.second - 1; y <= b.second + 1; ++y)
          {
            auto it = node_bins.find({x, y});
            if (it != node_bins.end())
              for (auto n : it->second)
                if (dot(subtract(nodes[n], p), subtract(nodes[n], p)) <=
                    tolerance * tolerance)
                  return n;
          }
        const auto n = nodes.size();
        nodes.push_back(p);
        node_bins[b].push_back(n);
        return n;
      };
      std::map<std::pair<std::size_t, std::size_t>, std::size_t> unique_edges;
      std::vector<bool>                                          inconsistent;
      for (auto s : contour)
      {
        const double norm    = std::sqrt(dot(s.normal, s.normal));
        const double length2 = dot(subtract(s.b, s.a), subtract(s.b, s.a));
        if (!(norm > 0.) || !std::isfinite(norm) || !(length2 > 0.) ||
            !std::isfinite(length2) || !std::isfinite(s.a.x) ||
            !std::isfinite(s.a.y))
          throw std::invalid_argument("Degenerate interface segment");
        s.normal = scale(s.normal, 1. / norm);
        auto a = node_id(s.a), b = node_id(s.b);
        if (a == b)
          continue;
        const std::pair<std::size_t, std::size_t> key{std::min(a, b),
                                                      std::max(a, b)};
        const auto existing = unique_edges.find(key);
        if (existing != unique_edges.end())
        {
          if (dot(s.normal, segments[existing->second].normal) < .5)
            inconsistent[existing->second] = true;
          continue;
        }
        unique_edges[key] = segments.size();
        segments.push_back(s);
        endpoints.push_back({{a, b}});
        inconsistent.push_back(false);
      }
      incident.resize(nodes.size());
      for (std::size_t i = 0; i < segments.size(); ++i)
        for (auto n : endpoints[i])
          incident[n].push_back(i);
      patch_normals.resize(segments.size());
      ambiguous_segment = inconsistent;
      for (std::size_t i = 0; i < segments.size(); ++i)
      {
        const auto &s = segments[i];
        Point       normal =
          scale(s.normal,
                std::sqrt(dot(subtract(s.b, s.a), subtract(s.b, s.a))));
        for (auto n : endpoints[i])
        {
          if (incident[n].size() > 2)
            ambiguous_segment[i] = true;
          for (auto j : incident[n])
            if (j != i && dot(s.normal, segments[j].normal) >= .5)
            {
              const auto &other = segments[j];
              const auto  edge  = subtract(other.b, other.a);
              normal =
                add(normal, scale(other.normal, std::sqrt(dot(edge, edge))));
            }
        }
        patch_normals[i] = scale(normal, 1. / std::sqrt(dot(normal, normal)));
      }
    }

    bool adjacent(std::size_t a, std::size_t b) const
    {
      for (auto n : endpoints[a])
        for (auto m : endpoints[b])
          if (n == m)
            return true;
      return false;
    }

    Bin bin(const Point &p) const
    {
      return {static_cast<int>(std::floor(p.x / radius)),
              static_cast<int>(std::floor(p.y / radius))};
    }

    std::vector<std::size_t> candidates(const Point &p) const
    {
      const auto               b = bin(p);
      std::vector<std::size_t> ids;
      for (int x = b.first - 1; x <= b.first + 1; ++x)
        for (int y = b.second - 1; y <= b.second + 1; ++y)
        {
          auto it = bins.find({x, y});
          if (it != bins.end())
            ids.insert(ids.end(), it->second.begin(), it->second.end());
        }
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      return ids;
    }

    Result measure(std::size_t source) const
    {
      Result result;
      result.status = Status::no_partner;
      if (ambiguous_segment[source])
      {
        result.status = Status::ambiguous;
        return result;
      }
      const auto &s         = segments[source];
      const Point origin    = scale(add(s.a, s.b), .5);
      const auto  nearby    = candidates(origin);
      double      gap       = std::numeric_limits<double>::infinity();
      bool        ambiguous = false;
      for (double sign : {-1., 1.})
      {
        if (pairing_phase != 0. && sign * pairing_phase < 0.)
          continue;
        const Point              direction = scale(patch_normals[source], sign);
        double                   first     = radius;
        std::vector<std::size_t> hits;
        for (auto i : nearby)
        {
          if (i == source)
            continue;
          const auto  &other       = segments[i];
          const Point  edge        = subtract(other.b, other.a);
          const double denominator = cross(direction, edge);
          if (std::abs(denominator) <= 1e-12 * std::sqrt(dot(edge, edge)))
            continue;
          const Point  offset = subtract(other.a, origin);
          const double t      = cross(offset, edge) / denominator;
          const double u      = cross(offset, direction) / denominator;
          if (t > 1e-9 * eps && t <= radius && t <= first + 1e-9 * eps &&
              u >= -1e-10 && u <= 1. + 1e-10)
          {
            if (t < first - 1e-9 * eps)
              hits.clear();
            first = std::min(first, t);
            hits.push_back(i);
          }
        }
        if (hits.empty())
          continue;
        // Do not skip a rejected foreground hit and look through it for a
        // more convenient partner. At a shared vertex check BOTH hit segments.
        bool accepted = true;
        for (auto i : hits)
          if (adjacent(source, i) || ambiguous_segment[i] ||
              dot(patch_normals[source], patch_normals[i]) > -.5)
            accepted = false;
        if (accepted)
        {
          if (first < gap)
            result.partner_direction = direction;
          gap = std::min(gap, first);
        }
        else
          ambiguous = true;
      }
      // Conservative: a competing ambiguous measurement prevents a reliable
      // label even if another ray found an admissible branch.
      if (ambiguous)
        result.status = Status::ambiguous;
      else if (std::isfinite(gap))
      {
        result.gap     = gap;
        result.overlap = 1. - std::tanh(gap / (2. * std::sqrt(2.) * eps));
        result.valid   = true;
        result.status  = Status::measured;
      }
      return result;
    }

    std::vector<Segment>                    segments;
    double                                  eps, radius;
    double                                  pairing_phase = 0.;
    std::map<Bin, std::vector<std::size_t>> bins;
    std::vector<std::array<std::size_t, 2>> endpoints;
    std::vector<std::vector<std::size_t>>   incident;
    std::vector<Point>                      patch_normals;
    std::vector<bool>                       ambiguous_segment;
    std::vector<Result>                     gaps;
  };
} // namespace CHNSProfileOverlap
#endif
