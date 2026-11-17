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
unchanged. The model currently requires transient fixed-mesh 2D `incompressible_chns`,
a Q1 tracer on axis-aligned rectangular cells, and positive constant mobility,
surface tension and interface thickness. Manufactured sources for this model,
3D, ALE, simplices and higher-order phase reconstruction are not implemented and
are rejected explicitly. Tree-based AMR is supported. The existing solver restriction
against combining checkpoint restart with tree-based AMR remains in place.

### Newton and geometry

Each time-step attempt uses **one ordinary Newton solve**. Geometry is prepared once
from the previous solution after the current prescribed boundary values have been
applied, preserving the exact timing of `v17_once`. This snapshot stays fixed during
Newton and finite-difference Jacobian checks. The local phase and velocity dependence
of K, its spatial gradient and the profile flux remain differentiated in the Jacobian.
There is no geometric fixed-point loop or successive nonlinear solves.

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
