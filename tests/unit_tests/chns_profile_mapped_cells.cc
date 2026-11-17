#include <chns_profile_geometry.h>

#include <fstream>
#include <iostream>

using CHNSProfileGeometry::CellMap;
using CHNSProfileOverlap::Point;
void check(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}

int main()
{
  try
  {
    // An affine phase on a curved P2 triangle is not affine in physical space.
    // Explicit polynomial map and chain-rule gradients are independent oracles.
    for (bool simplex : {true, false})
      for (unsigned degree : {1u, 2u})
      {
        CellMap map;
        map.simplex = simplex;
        map.degree  = degree;
        for (unsigned k = 0; k < map.size(); ++k)
        {
          const auto u = map.support_point(k);
          map.nodes[k] = {2. + 1.2 * u.x + .3 * u.y +
                            (degree == 2 ? .2 * u.x * u.y : 0.),
                          -1. + .2 * u.x + .9 * u.y +
                            (degree == 2 ? .15 * u.x * u.x : 0.)};
        }
        const auto             bounds = map.bounds();
        CHNSProfileWidth::Cell cell{bounds.first,
                                    bounds.second,
                                    {{-.7, .1, -.5, .3}}};
        cell.mapped  = true;
        cell.mapping = map;
        const Point physical{2.33 + (degree == 2 ? .012 : 0.),
                             -.69 + (degree == 2 ? .006 : 0.)};
        Point       unit, gradient;
        check(map.inverse(physical, unit), "Mapped inverse failed");
        check(std::abs(unit.x - .2) + std::abs(unit.y - .3) < 1e-12,
              "Wrong inverse mapping");
        check(std::abs(cell.evaluate(physical, &gradient) + .48) < 1e-12,
              "Wrong P1/Q1 phase");
        const double a = degree == 2 ? 1.26 : 1.2, b = degree == 2 ? .34 : .3,
                     c = degree == 2 ? .26 : .2, determinant = a * .9 - b * c;
        check(std::abs(gradient.x - (.72 - .2 * c) / determinant) < 1e-12 &&
                std::abs(gradient.y - (-.8 * b + .2 * a) / determinant) < 1e-12,
              "Wrong physical phase gradient");
        check(cell.contains(physical), "Lost interior mapped point");
        if (simplex)
          check(!cell.contains(map.evaluate({.8, .8})),
                "Triangle search accepted a point only in its box");
      }

    {
      CellMap translated;
      translated.simplex  = true;
      translated.nodes[0] = {2., 3.};
      translated.nodes[1] = {2. + 1e-7, 3.};
      translated.nodes[2] = {2., 3. + 1e-7};
      Point unit;
      check(translated.inverse({2. + .37e-7, 3. + .28e-7}, unit),
            "Inverse mapping rejected a small translated cell");
      for (const Point p : {Point{0., 0.},
                            Point{1., 0.},
                            Point{0., 1.},
                            Point{.5, 0.},
                            Point{0., .5},
                            Point{.5, .5}})
        check(translated.contains(translated.evaluate(p)),
              "Small translated cell rejected its own vertex/edge");
      check(!translated.contains(translated.evaluate({-.01, .5})),
            "Translated-cell tolerance accepted an exterior point");
    }

    // Rotation/translation must preserve width feedback, even with two cells
    // sharing every bounding-box coordinate and distributed anchor measures.
    std::array<CHNSProfileWidth::Value, 2> result;
    for (unsigned rotated = 0; rotated < 2; ++rotated)
    {
      const auto transform = [&](Point p) {
        return rotated ?
                 Point{.6 * p.x - .8 * p.y + 2., .8 * p.x + .6 * p.y - 1.} :
                 p;
      };
      std::vector<CHNSProfileWidth::Cell> cells;
      for (unsigned i = 0; i < 80; ++i)
        for (unsigned j = 0; j < 12; ++j)
          for (unsigned side = 0; side < 2; ++side)
          {
            const double               x = -1. + .025 * i, y = -.3 + .05 * j;
            const std::array<Point, 3> p =
              side == 0 ?
                std::array<Point, 3>{{{x, y}, {x + .025, y}, {x, y + .05}}} :
                std::array<Point, 3>{
                  {{x + .025, y + .05}, {x, y + .05}, {x + .025, y}}};
            CellMap map;
            map.simplex = true;
            std::array<double, 4> phase{};
            for (unsigned k = 0; k < 3; ++k)
            {
              map.nodes[k] = transform(p[k]);
              phase[k]     = std::tanh(p[k].x / (1.2 * std::sqrt(2.) * .08));
            }
            phase[3]                 = phase[2];
            const auto             b = map.bounds();
            CHNSProfileWidth::Cell cell{b.first, b.second, phase};
            cell.mapped  = true;
            cell.mapping = map;
            cells.push_back(cell);
          }
      CHNSProfileWidth::Geometry geometry(std::move(cells), .08);
      check(geometry.size() > 0 && geometry.max_anchor_residual() < 1e-10,
            "Mapped zero contour");
      const auto zero = [](Point) { return 0.; };
      geometry.install({geometry.measure(0, 2, zero),
                        geometry.measure(1, 2, zero)},
                       2.);
      result[rotated] = geometry.evaluate(transform({.073, .017}));
      check(result[rotated].gain > 1.03 && result[rotated].confidence > .99,
            "Mapped width lost tails");
    }
    check(std::abs(result[0].gain - result[1].gain) < 1e-10 &&
            std::abs(result[0].tail - result[1].tail) < 1e-10,
          "Width not invariant under rigid motion");

    {
      CHNSProfileOverlap::ReflectingBox box;
      box.lower            = {0., -1.};
      box.upper            = {1., 1.};
      box.reflect_lower[0] = true;
      const std::vector<CHNSProfileOverlap::MidlineSource> source{
        {{.1, -.5}, {.1, .5}, .03}};
      CHNSProfileOverlap::SmoothPenalty straight(source,
                                                 .1,
                                                 std::sqrt(2.),
                                                 box);
      const auto                        expected = straight.evaluate({0., .1});
      box.axis                                   = {.6, .8};
      auto rotated                               = source;
      for (auto &line : rotated)
      {
        line.a = box.physical(line.a);
        line.b = box.physical(line.b);
      }
      CHNSProfileOverlap::SmoothPenalty penalty(rotated,
                                                .1,
                                                std::sqrt(2.),
                                                box);
      const auto actual = penalty.evaluate(box.physical({0., .1}));
      check(std::abs(actual.value - expected.value) < 1e-12 &&
              std::abs(CHNSProfileOverlap::dot(actual.gradient, box.axis)) <
                1e-12,
            "Natural-boundary reflection not invariant under rotation");
    }

    // Frozen geometry scalar follows reference coordinates; its covector
    // transforms with J_current^{-T} J_snapshot^T.
    CHNSProfileGeometry::Fields fields;
    CellMap                     map;
    map.simplex                    = true;
    map.nodes[0]                   = {0., 0.};
    map.nodes[1]                   = {1., 0.};
    map.nodes[2]                   = {0., 1.};
    fields.reference_cells["cell"] = map;
    fields.overlap = std::make_shared<CHNSProfileOverlap::SmoothPenalty>(
      std::vector<CHNSProfileOverlap::MidlineSource>{
        {{.1, -.5}, {.1, .5}, .03}},
      .1,
      std::sqrt(2.));
    const auto old = fields.evaluate(Point{.13, .07});
    const std::array<std::array<double, 2>, 2> inverse{
      {{{.5, -1. / 6.}}, {{0., 2. / 3.}}}};
    const auto current = fields.evaluate_on_cell("cell", {.13, .07}, inverse);
    check(current.value == old.value &&
            std::abs(current.gradient.x - .5 * old.gradient.x) < 1e-12 &&
            std::abs(current.gradient.y -
                     (-old.gradient.x / 6. + 2. * old.gradient.y / 3.)) < 1e-12,
          "Frozen geometry covector transport");
    std::ofstream("output") << "Mapped P1/Q1 cells, curved geometry, width "
                               "objectivity and covector transport OK\n";
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
