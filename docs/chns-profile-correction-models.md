# Profile correction models

The profile correction parameters are grouped under `Cahn Hilliard/profile correction`.
Move the old flat correction entries into this subsection. `enable flux correction`
remains directly under `Cahn Hilliard`.

## Historical constant model

```text
subsection Cahn Hilliard
  subsection profile correction
    set enable profile correction = true
    set profile correction model = constant
    set profile correction strength = 100
  end
end
```

This preserves the historical implicit operator, nodal reconstruction, regularization
and tail cutoff. Its coefficient is `2 * strength * M_max * sigma_tilde / epsilon`,
where `sigma_tilde = 3 sigma / (2 sqrt(2))`. Variable mobility retains the historical
global maximum and its frozen treatment in the Jacobian. The model defaults to
`constant`, the strength defaults to `0.6`, and correction is disabled by default.
This model remains available in CHNS, CHNS-ALE and enlarged CHNS-ALE.

## Local timescale model

```text
subsection Cahn Hilliard
  subsection mobility
    set model = constant
    set constant value = 2e-9
  end
  set surface tension representation = laplace beltrami
  subsection profile correction
    set enable profile correction = true
    set profile correction model = local timescale
    set profile correction timescale factor = 50
  end
end
```

This integrates the validated `v17_once` law. No environment variable or dedicated
experimental executable is required. `profile correction strength` is unused in
this model; `profile correction timescale factor` is the dimensionless factor C
(default 50). C is not numerically equivalent to the historical strength.

Let `w = grad(I_h(atanh_regularized(phi)))`, `b = M sigma_tilde / epsilon`,
`z = w . grad(u) w / (|w|^2 + beta^2/(2 epsilon^2))`, with `beta = 0.05`.
The timescale coefficient is

```text
K_rate = C [b + epsilon^2 (sqrt(z^2 + delta^2) - delta)]
delta  = 1e-8 b / epsilon^2
```

The implementation evaluates the square-root difference in its cancellation-free
form. The validated geometric width feedback multiplies this coefficient, using
normal distances from phi=0 to the 0.9, 0.99, 0.999 and 0.9999 contours. Only
complete, sufficiently saturated rays contribute; overlap reduces their confidence.
The width gain and its support are smoothly distributed along and across the
interface, as in the experimental version.

A smooth overlap penalty G is built from facing +/-0.99 contours and their
zero-equivalent separation. It attenuates the coefficient inside the interacting
film. With `q = max(2 epsilon^2 |w|^2 - 1, 0)` and `r = q^3/(1+q^3)`,

```text
K = K_width [1/(1+G) + (1 - 1/(1+G)) r]
```

This preserves the validated recovery of useful correction on compressed profiles.
The overlap range, natural-boundary reflection, beta and physical tail cutoff are
unchanged. The model supports transient 2D `incompressible_chns`,
`incompressible_chns_ale` and `incompressible_chns_ale_enlarged`, with a P1 tracer
on triangles or a Q1 tracer on quadrilaterals. Position mappings may be linear or
quadratic, including curved cells. Positive constant mobility, surface tension
and interface thickness are required. Manufactured sources for this model, 3D,
higher-order phase reconstruction and position mappings above degree two are
rejected explicitly. Mesh adaptation remains subject to each solver's existing
capabilities; the restriction against combining checkpoint restart with tree-based
AMR remains in place.

### Newton and geometry

Each time-step attempt uses **one ordinary Newton solve**. Geometry is prepared once
from the previous solution after the current prescribed boundary values have been
applied, preserving the exact timing of `v17_once`. This snapshot stays fixed during
Newton and finite-difference Jacobian checks. The local phase and velocity dependence
of K, its spatial gradient and the profile flux remain differentiated in the Jacobian.
There is no geometric fixed-point loop or successive nonlinear solves.

In ALE, the geometric snapshot is attached to the reference cells for the duration
of the attempt. At reference point `xi`, a scalar geometric indicator is evaluated
at `X_snapshot(xi)`. Its physical gradient is transformed by
`J_current^{-T} J_snapshot^T`. Thus a mesh increment with physical gradient `G`
contributes `-G^T grad(indicator)` to the linearization, while the indicator's
scalar value remains frozen. These terms are included in the SUPG strong residual
and its Jacobian. Phase and velocity remain implicit. Physical distances and
interaction partners are recomputed at the next attempt; there is no within-step
geometric iteration. This is a lagged geometric model, not an assertion that
mesh motion leaves all local physical gradients unchanged.

Width rays use the transported P1/Q1 phase in the physical isoparametric cell,
with true cell containment rather than containment in its bounding box. The
original axis-aligned fixed-mesh Q1 arithmetic is retained. Kernel reflection at
straight natural boundaries uses their physical frame, including rotated rectangular
domains; curved boundaries are not reflected through a fictitious chord plane.
Enlarged ALE applies this correction to phi; the additional psi field does not
replace phi in the profile reconstruction or the width measurements.

Accepted-step exports use the same snapshot as the solve. Initial/restart output
builds a snapshot when needed. AMR and rejected attempts rebuild geometry before
the next solve; each solver owns its own fields, with no mutable process globals.

## ParaView export

```text
subsection Postprocessing
  subsection profile correction
    set enable = true
  end
end
```

Both volume and skin exports use the same nodal inverse-profile reconstruction
as assembly:

- `profile_correction_strength`: historical strength in constant mode; equivalent
  local strength `K/(2b)` in local mode.
- `profile_correction_coefficient`: actual K used in the profile flux.
- `profile_correction_flux`: correction flux vector.

Local mode additionally exports `profile_correction_geometric_penalty`,
`profile_correction_retained_fraction`, `profile_correction_feedback_gain`,
`profile_correction_feedback_support`, `profile_correction_width_error`,
`profile_correction_feedback_confidence` and `profile_correction_timescale_strength`.
These exports do not trigger another nonlinear solve.

## Verification

Focused tests cover nested parameter parsing, invalid mobility, historical constant
behavior, the local coefficient and its spatial/Newton mixed derivatives, all four
SUPG combinations with LB and with/without chemical flux projection, disappearing
width-measurement tails, overlap-kernel gradients, boundary reflection, and volume/skin
exports. Numerical integration checks use the native Ubuntu BLAS preload, including
bounded comparisons with `v17_once` in serial, MPI and restart configurations,
and an AMR smoke test on 24 ranks. Exported fields match the old executable in
all of those direct comparisons. The narrow-film example retains pre-existing
differences between serial and MPI diagnostics, and between uninterrupted BDF2
and checkpoint restart (maximum phase difference about 9e-5). This integration
does not change the time-history manager or claim to fix those baseline differences.

The ALE extension adds checks for P1/Q1 interpolation on linear and quadratic
position maps, actual triangle containment, small translated cells, rotated width
measurements, and reflection at straight walls despite curved interior cells.
Finite differences check the transport of nonzero overlap and width-feedback
gradients, including the mesh contribution to the coefficient's spatial gradient.
Volume and skin exports are checked on a displaced triangular mesh.

Assembled ALE and enlarged-ALE Jacobians are checked on triangles with P2 mesh
positions, all SUPG combinations, LB, and optional chemical flux projection.
The complete matrix is checked on a profile with a resolved normal. The original
polynomial case also checks the phase-equation rows, including its nearly constant
cell: the existing LB stress is nonsmooth at zero phase gradient, so its momentum
derivative there does not admit the same classical finite-difference check. This
behavior is reproduced with correction disabled; neither LB nor the Jacobian
test tolerances were changed for this extension.

Short moving-film runs exercise nonzero overlap attenuation in both ALE solvers
on 2 and 24 ranks. Active width feedback under mesh motion is checked separately
by the coefficient derivative test; it is not activated in that short film fixture.
