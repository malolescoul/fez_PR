#ifndef CHNS_PROFILE_WIDTH_H
#define CHNS_PROFILE_WIDTH_H

#include <chns_profile_overlap.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

// Width feedback from the transported Q1 phase. Immutable during Newton;
// rebuilt once per attempt from the previous solution with current boundary
// values.
namespace CHNSProfileWidth
{
  using CHNSProfileOverlap::add;
  using CHNSProfileOverlap::dot;
  using CHNSProfileOverlap::Point;
  using CHNSProfileOverlap::scale;
  using CHNSProfileOverlap::Segment;
  using CHNSProfileOverlap::subtract;
  inline constexpr std::array<double, 4> levels{{.9, .99, .999, .9999}};
  inline double                          smoothstep(double t)
  {
    const double z = std::clamp(t, 0., 1.);
    return z * z * z * (10. + z * (-15. + 6. * z));
  }
  struct Measurement
  {
    bool   valid         = false;
    double error_squared = 0., tail_distance = INFINITY, confidence = 0.;
  };
  inline Measurement measure_ray(const std::vector<double> &d,
                                 const std::vector<double> &p,
                                 double                     eps,
                                 const std::vector<double> &overlap = {})
  {
    if (!(eps > 0.) || d.size() != p.size() ||
        (!overlap.empty() && overlap.size() != p.size()))
      throw std::invalid_argument("Invalid feedback ray");
    std::size_t stop = 0;
    while (stop < p.size() && std::isfinite(p[stop]) && p[stop] >= -1e-6)
      ++stop;
    if (stop < 2)
      return {};
    Measurement m;
    std::size_t last = 0;
    for (double alpha : levels)
    {
      std::size_t k = 0;
      while (k < stop && p[k] < alpha)
        ++k;
      if (k == 0 || k == stop)
        return {};
      const double distance =
        d[k - 1] + (d[k] - d[k - 1]) * (alpha - p[k - 1]) / (p[k] - p[k - 1]);
      const double error =
        distance / (std::sqrt(2.) * eps * std::atanh(alpha)) - 1.;
      m.error_squared += error * error / 4.;
      m.tail_distance = distance;
      last            = k;
    }
    m.confidence = smoothstep(
      (*std::max_element(p.begin(), p.begin() + stop) - levels.back()) /
      (1. - levels.back()));
    if (!overlap.empty())
      m.confidence /=
        1. + std::max(0.,
                      *std::max_element(overlap.begin(),
                                        overlap.begin() + last + 1));
    m.error_squared *= m.confidence * m.confidence;
    m.valid = true;
    return m;
  }
  // Return support and derivative with respect to unsigned geometric distance.
  inline std::pair<double, double>
  support(double distance, double tail, double eps, double confidence)
  {
    if (!(eps > 0.))
      throw std::invalid_argument("Invalid feedback epsilon");
    const double z = confidence * std::max(std::abs(distance) - tail, 0.) / eps;
    if (z <= 0.)
      return {1., 0.};
    if (z >= 1.)
      return {0., 0.};
    return {1. - smoothstep(z),
            -30. * z * z * (z - 1.) * (z - 1.) * confidence / eps};
  }
  struct Value
  {
    double multiplier = 1.;
    Point  gradient;
    double gain = 1., support = 1., error = 0., confidence = 0.,
           tail = INFINITY, distance = INFINITY;
  };
  struct Cell
  {
    Point                 lower, upper;
    std::array<double, 4> phase; // 00,10,01,11
    double                evaluate(Point p, Point *gradient = nullptr) const
    {
      const double dx = upper.x - lower.x, dy = upper.y - lower.y;
      const double x = (p.x - lower.x) / dx, y = (p.y - lower.y) / dy;
      if (gradient)
        *gradient = {
          ((1 - y) * (phase[1] - phase[0]) + y * (phase[3] - phase[2])) / dx,
          ((1 - x) * (phase[2] - phase[0]) + x * (phase[3] - phase[1])) / dy};
      return (1 - x) * (1 - y) * phase[0] + x * (1 - y) * phase[1] +
             (1 - x) * y * phase[2] + x * y * phase[3];
    }
  };
  struct Box
  {
    Point lo, hi;
  };
  inline double box_distance(Point p, const Box &b)
  {
    const double x = std::max({b.lo.x - p.x, 0., p.x - b.hi.x});
    const double y = std::max({b.lo.y - p.y, 0., p.y - b.hi.y});
    return x * x + y * y;
  }
  class Tree
  {
    struct Node
    {
      Box         box;
      std::size_t begin, end;
      int         left = -1, right = -1;
    };
    std::vector<Box>         boxes;
    std::vector<std::size_t> order;
    std::vector<Node>        nodes;
    int                      build(std::size_t begin, std::size_t end)
    {
      Box b{{INFINITY, INFINITY}, {-INFINITY, -INFINITY}};
      for (std::size_t k = begin; k < end; ++k)
      {
        const auto &a = boxes[order[k]];
        b.lo.x        = std::min(b.lo.x, a.lo.x);
        b.lo.y        = std::min(b.lo.y, a.lo.y);
        b.hi.x        = std::max(b.hi.x, a.hi.x);
        b.hi.y        = std::max(b.hi.y, a.hi.y);
      }
      const int id = nodes.size();
      nodes.push_back({b, begin, end});
      if (end - begin > 8)
      {
        const unsigned axis = (b.hi.y - b.lo.y > b.hi.x - b.lo.x);
        const auto     mid  = (begin + end) / 2;
        std::nth_element(order.begin() + begin,
                         order.begin() + mid,
                         order.begin() + end,
                         [&](auto i, auto j) {
                           const double a =
                                          boxes[i].lo[axis] + boxes[i].hi[axis],
                                        c =
                                          boxes[j].lo[axis] + boxes[j].hi[axis];
                           return a == c ? i < j : a < c;
                         });
        const int left = build(begin, mid), right = build(mid, end);
        nodes[id].left  = left;
        nodes[id].right = right;
      }
      return id;
    }
    template <class Distance>
    void search(int             n,
                Point           p,
                double         &best,
                std::size_t    &selected,
                const Distance &distance) const
    {
      const auto &a = nodes[n];
      if (box_distance(p, a.box) > best)
        return;
      if (a.left < 0)
      {
        for (auto k = a.begin; k < a.end; ++k)
        {
          auto         i = order[k];
          const double d = distance(i);
          if (d < best || (d == best && i < selected))
          {
            best     = d;
            selected = i;
          }
        }
        return;
      }
      int first = a.left, second = a.right;
      if (box_distance(p, nodes[first].box) >
          box_distance(p, nodes[second].box))
        std::swap(first, second);
      search(first, p, best, selected, distance);
      search(second, p, best, selected, distance);
    }

  public:
    Tree() = default;
    explicit Tree(std::vector<Box> b)
      : boxes(std::move(b))
    {
      order.resize(boxes.size());
      std::iota(order.begin(), order.end(), 0);
      if (!order.empty())
        build(0, order.size());
    }
    template <class Distance>
    std::size_t nearest(Point p, const Distance &distance, double &best) const
    {
      std::size_t selected = boxes.size();
      if (!nodes.empty())
        search(0, p, best, selected, distance);
      return selected;
    }
  };
  class Geometry
  {
    struct Anchor
    {
      Segment                    edge;
      Point                      p;
      double                     length = 0., error2 = 0., smoothed = 0.;
      std::array<Measurement, 2> flank;
    };
    std::vector<Cell>   cells;
    double              eps, C = 0., anchor_residual = 0.;
    Tree                cell_tree, edge_tree;
    std::vector<Anchor> anchors;
    struct Key
    {
      std::uint64_t x, y;
      bool operator==(const Key &b) const { return x == b.x && y == b.y; }
    };
    struct Hash
    {
      std::size_t operator()(const Key &k) const
      {
        return k.x ^ (k.y + 0x9e3779b97f4a7c15ULL + (k.x << 6) + (k.x >> 2));
      }
    };
    mutable std::mutex                           cache_mutex;
    mutable std::unordered_map<Key, Value, Hash> cache;
    bool sample(Point p, double &phi, Point *gradient = nullptr) const
    {
      double     best = 1e-20 * eps * eps;
      const auto i    = cell_tree.nearest(
        p,
        [&](auto k) {
          return box_distance(p, {cells[k].lower, cells[k].upper});
        },
        best);
      if (i == cells.size())
        return false;
      phi = cells[i].evaluate(p, gradient);
      return true;
    }
    static Point closest(Point p, const Segment &s)
    {
      const auto ab = subtract(s.b, s.a);
      return add(
        s.a,
        scale(ab, std::clamp(dot(subtract(p, s.a), ab) / dot(ab, ab), 0., 1.)));
    }
    void extract()
    {
      std::map<std::array<long long, 4>, bool> seen;
      for (const auto &cell : cells)
      {
        if (*std::min_element(cell.phase.begin(), cell.phase.end()) > 0. ||
            *std::max_element(cell.phase.begin(), cell.phase.end()) < 0.)
          continue;
        // Match the 2-subdivision output used by the saved-state prototype.
        for (unsigned i = 0; i < 2; ++i)
          for (unsigned j = 0; j < 2; ++j)
          {
            std::array<Point, 4> p;
            for (unsigned k = 0; k < 4; ++k)
              p[k] = {cell.lower.x +
                        (i + (k % 2)) * .5 * (cell.upper.x - cell.lower.x),
                      cell.lower.y +
                        (j + (k / 2)) * .5 * (cell.upper.y - cell.lower.y)};
            for (const std::array<unsigned, 3> tri :
                 {std::array<unsigned, 3>{{0, 1, 2}}, {{1, 3, 2}}})
            {
              std::array<double, 3> f;
              for (unsigned k = 0; k < 3; ++k)
                f[k] = cell.evaluate(p[tri[k]]);
              if (f[0] == 0. && f[1] == 0. && f[2] == 0.)
                continue;
              std::vector<Point> roots;
              const auto         append = [&](Point q) {
                for (auto r : roots)
                  if (dot(subtract(q, r), subtract(q, r)) < 1e-24 * eps * eps)
                    return;
                roots.push_back(q);
              };
              for (unsigned k = 0; k < 3; ++k)
              {
                unsigned l = (k + 1) % 3;
                if (f[k] == 0.)
                  append(p[tri[k]]);
                if (f[k] * f[l] < 0.)
                  append(add(p[tri[k]],
                             scale(subtract(p[tri[l]], p[tri[k]]),
                                   -f[k] / (f[l] - f[k]))));
              }
              if (roots.size() != 2)
                continue;
              auto a = roots[0], b = roots[1];
              if (std::make_pair(a.x, a.y) > std::make_pair(b.x, b.y))
                std::swap(a, b);
              std::array<long long, 4> key{{std::llround(a.x / (1e-8 * eps)),
                                            std::llround(a.y / (1e-8 * eps)),
                                            std::llround(b.x / (1e-8 * eps)),
                                            std::llround(b.y / (1e-8 * eps))}};
              if (seen.count(key))
                continue;
              seen[key]           = true;
              const auto   ab     = subtract(b, a);
              const double length = std::sqrt(dot(ab, ab));
              if (length <= 1e-10 * eps)
                continue;
              Point       midpoint          = scale(add(a, b), .5), gradient;
              const Point original_midpoint = midpoint;
              cell.evaluate(midpoint, &gradient);
              Point normal = scale({-ab.y, ab.x}, 1. / length);
              if (dot(normal, gradient) < 0.)
                normal = scale(normal, -1.);
              for (unsigned it = 0; it < 8; ++it)
              {
                double f0;
                Point  g;
                f0 = cell.evaluate(midpoint, &g);
                if (std::abs(f0) <= 1e-10)
                  break;
                const double derivative = dot(g, normal);
                if (std::abs(derivative) < 1e-12)
                  break;
                const auto next =
                  subtract(midpoint, scale(normal, f0 / derivative));
                if (box_distance(next, {cell.lower, cell.upper}) > 0.)
                  break;
                midpoint = next;
              }
              double f0 = cell.evaluate(midpoint);
              if (std::abs(f0) > 1e-10)
              {
                // A Q1 saddle can make a chord-normal Newton step leave the
                // cell. Fall back to exact linear roots on axis-aligned cuts
                // through the original midpoint and the four cell edges.
                // This changes only anchor localization, never the S0 law.
                double     closest2 = INFINITY;
                Point      best;
                const auto root = [&](Point lo, Point hi) {
                  const double u = cell.evaluate(lo), v = cell.evaluate(hi);
                  if ((u > 0. && v > 0.) || (u < 0. && v < 0.))
                    return;
                  const auto point =
                    add(lo,
                        scale(subtract(hi, lo),
                              u == v ? .5 : std::clamp(u / (u - v), 0., 1.)));
                  const auto d = subtract(point, original_midpoint);
                  if (dot(d, d) < closest2)
                  {
                    closest2 = dot(d, d);
                    best     = point;
                  }
                };
                for (double y :
                     {cell.lower.y, original_midpoint.y, cell.upper.y})
                  root({cell.lower.x, y}, {cell.upper.x, y});
                for (double x :
                     {cell.lower.x, original_midpoint.x, cell.upper.x})
                  root({x, cell.lower.y}, {x, cell.upper.y});
                if (!std::isfinite(closest2))
                  throw std::runtime_error("Local profile correction Q1 zero "
                                           "projection found no bracket");
                midpoint = best;
                f0       = cell.evaluate(midpoint);
              }
              if (std::abs(f0) > 1e-8)
                throw std::runtime_error(
                  "Local profile correction Q1 zero projection failed");
              anchor_residual = std::max(anchor_residual, std::abs(f0));
              Anchor anchor;
              anchor.edge   = {a, b, normal};
              anchor.p      = midpoint;
              anchor.length = length;
              anchors.push_back(anchor);
            }
          }
      }
      std::vector<Box> boxes;
      for (const auto &a : anchors)
        boxes.push_back(
          {{std::min(a.edge.a.x, a.edge.b.x), std::min(a.edge.a.y, a.edge.b.y)},
           {std::max(a.edge.a.x, a.edge.b.x),
            std::max(a.edge.a.y, a.edge.b.y)}});
      edge_tree = Tree(std::move(boxes));
    }
    Value evaluate_uncached(Point p) const
    {
      if (anchors.empty())
        return {};
      double d2 = INFINITY;
      auto   i  = edge_tree.nearest(
        p,
        [&](auto k) {
          auto d = subtract(p, closest(p, anchors[k].edge));
          return dot(d, d);
        },
        d2);
      const auto    &a        = anchors[i];
      auto           d        = subtract(p, closest(p, a.edge));
      const double   distance = std::sqrt(d2);
      const unsigned side     = dot(d, a.edge.normal) >= 0.;
      const auto    &m        = a.flank[side];
      const double   A        = std::sqrt(1. + C * C * a.smoothed);
      const auto     B = support(distance, m.tail_distance, eps, m.confidence);
      return {A * B.first,
              distance > 0. ? scale(d, A * B.second / distance) : Point{},
              A,
              B.first,
              std::sqrt(a.smoothed),
              m.confidence,
              m.tail_distance,
              distance};
    }

  public:
    Geometry(std::vector<Cell> input, double epsilon)
      : cells(std::move(input))
      , eps(epsilon)
    {
      if (!(eps > 0.))
        throw std::invalid_argument(
          "Local profile correction epsilon must be positive");
      // Determinism independent of MPI ownership and repartitioning.
      std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
        return std::tie(a.lower.x, a.lower.y, a.upper.x, a.upper.y) <
               std::tie(b.lower.x, b.lower.y, b.upper.x, b.upper.y);
      });
      std::vector<Box> boxes;
      for (const auto &c : cells)
      {
        if (!(c.upper.x > c.lower.x && c.upper.y > c.lower.y))
          throw std::invalid_argument("Invalid feedback cell");
        boxes.push_back({c.lower, c.upper});
      }
      cell_tree = Tree(std::move(boxes));
      extract();
    }
    std::size_t size() const { return anchors.size(); }
    double      max_anchor_residual() const { return anchor_residual; }
    // Each rank measures its subset; all ranks install the same gathered data.
    template <class Penalty>
    std::vector<double>
    measure(unsigned rank, unsigned ranks, const Penalty &penalty) const
    {
      if (ranks == 0 || rank >= ranks)
        throw std::invalid_argument("Invalid feedback partition");
      std::vector<double> packed;
      for (std::size_t i = rank; i < anchors.size(); i += ranks)
      {
        const auto                &a = anchors[i];
        std::array<Measurement, 2> m;
        for (unsigned side = 0; side < 2; ++side)
        {
          std::vector<double> d, p, g;
          for (unsigned k = 0; k <= 256; ++k)
          {
            const double distance = k * eps / 16.;
            const double sign     = side ? 1. : -1.;
            const Point  x = add(a.p, scale(a.edge.normal, sign * distance));
            double       phi;
            if (!sample(x, phi))
              break;
            d.push_back(distance);
            p.push_back(sign * phi);
            g.push_back(penalty(x));
            if (sign * phi < -1e-6)
              break;
          }
          m[side] = measure_ray(d, p, eps, g);
        }
        packed.insert(packed.end(),
                      {double(i),
                       .5 * (m[0].error_squared + m[1].error_squared),
                       m[0].tail_distance,
                       m[1].tail_distance,
                       m[0].confidence,
                       m[1].confidence});
      }
      return packed;
    }
    void install(const std::vector<std::vector<double>> &parts, double factor)
    {
      C = factor;
      if (!(C >= 0.) || !std::isfinite(C))
        throw std::invalid_argument("Invalid Local profile correction C");
      std::vector<bool> seen(anchors.size());
      for (const auto &part : parts)
      {
        if (part.size() % 6)
          throw std::invalid_argument("Invalid feedback record");
        for (std::size_t k = 0; k < part.size(); k += 6)
        {
          const auto i = static_cast<std::size_t>(part[k]);
          if (i >= anchors.size() || seen[i])
            throw std::invalid_argument("Duplicate/invalid feedback anchor");
          seen[i]  = true;
          auto &a  = anchors[i];
          a.error2 = part[k + 1];
          for (unsigned side = 0; side < 2; ++side)
          {
            a.flank[side].tail_distance = part[k + 2 + side];
            a.flank[side].confidence    = part[k + 4 + side];
            a.flank[side].valid = std::isfinite(a.flank[side].tail_distance);
          }
        }
      }
      if (std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::invalid_argument("Missing feedback anchor");
      using Bin = std::pair<int, int>;
      std::map<Bin, std::vector<std::size_t>> bins;
      const auto                              bin = [&](Point p) {
        return Bin{int(std::floor(p.x / (3. * eps))),
                   int(std::floor(p.y / (3. * eps)))};
      };
      for (std::size_t i = 0; i < anchors.size(); ++i)
        bins[bin(anchors[i].p)].push_back(i);
      for (auto &a : anchors)
      {
        const auto b      = bin(a.p);
        double     weight = 0., sum = 0.;
        for (int x = b.first - 1; x <= b.first + 1; ++x)
          for (int y = b.second - 1; y <= b.second + 1; ++y)
          {
            auto it = bins.find({x, y});
            if (it == bins.end())
              continue;
            for (auto j : it->second)
            {
              const auto  &other = anchors[j];
              const auto   d     = subtract(a.p, other.p);
              const double r2    = dot(d, d);
              if (r2 > 9. * eps * eps ||
                  dot(a.edge.normal, other.edge.normal) <= .5)
                continue;
              const double w = other.length * std::exp(-.5 * r2 / (eps * eps));
              sum += w * other.error2;
              weight += w;
            }
          }
        a.smoothed = weight > 0. ? sum / weight : 0.;
      }
      std::lock_guard<std::mutex> lock(cache_mutex);
      cache.clear();
    }
    Value evaluate(Point p) const
    {
      Key key;
      std::memcpy(&key.x, &p.x, sizeof(double));
      std::memcpy(&key.y, &p.y, sizeof(double));
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto                        it = cache.find(key);
        if (it != cache.end())
          return it->second;
      }
      auto value = evaluate_uncached(p);
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(key, value);
      }
      return value;
    }
  };
} // namespace CHNSProfileWidth
#endif
