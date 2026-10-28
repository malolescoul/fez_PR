# Audit against the reference energy branch

Reference: `arthurbawin/fez`, `chns_brunk_ten_eikelder_energy_stable`, inspected
at `f626f7c7ee58fea4777b6bf4d0a0a162f8777283`. The comparison concerns its
quasi-incompressible energy assembler, not its independent Stepien model.
This is an implementation audit of that snapshot, not a claim about later
remote changes.

## Identical temporal ingredients

Let `rho+ = rho(clip(phi,-1,1))`, `beta=(rho0-rho1)/2`, and
`F(phi)=(phi^2-1)^2/4`. Both implementations use, for BDF1,

```
I = [ (rho+ - rho+_old) u / 2 + rho+_old (u-u_old) ] / dt
Fbar' = (phi+phi_old)(phi^2+phi_old^2-2)/4.
```

The polynomial divided difference remains regular when the two phases coincide.
Its derivative and the inertia derivatives are identical to the reference.
For BDF2 both use `sqrt(rho+) D_BDF(sqrt(rho+) u)` and the current cubic
`phi*(phi^2-1)`. As in the source, BDF2 requires a BDF1 startup with a start-step ratio strictly
between zero and one. Only the third BDF coefficient determines whether the current
step is BDF2: startup can use BDF1 with two allocated history vectors.

The reference's BDF-history fix (`87666fd`) is also carried over: inactive
coefficients are cleared and restarting with a new timestep preserves the last
accepted timestep. Its unrelated pressure-gauge changes are not included.

## Differences required by Abels

Write `Q=M grad(mu)+C`, with the existing implicit reconstructed correction `C`,
and `J=-beta Q`. We retain `div(u)=0`, diffusion driven by `grad(mu)`, and the full
Abels stress `2 eta D(u)`. We do not import the quasi-incompressible pressure
coupling, `grad(mu+alpha p)`, or its additional volumetric viscous term.

On a fixed mesh the momentum transport is

```
1/2 ((rho u+J).grad u, v) - 1/2 ((rho u+J).grad v, u),
```

and the phase equation is

```
(D_BDF phi, q) - (phi u, grad q) + (Q, grad q) + (source_phi,q) = 0.
```

As in the reference, affine density remains in transport and gravity; positive
kinetic density and viscosity are evaluated from clipped phase. No solution
DOFs are clipped. Limiter derivatives vanish outside [-1,1]; at the kinks the
chosen one-sided convention is not a claim of differentiability.

## ALE and enlarged ALE

Set `a=u-w`, `m=rho a+J`. The weak momentum transport becomes

```
1/2 (grad(u)m,v) - 1/2 (grad(v)m,u) + 1/2 (rho div(w) u,v).
```

The weak phase transport becomes `-(phi a,grad q)+(phi div(w),q)`.
Consequently the strong residual additions are

```
R_momentum: 1/2 [ beta grad(phi).a + rho div(u) + div(J) ] u
R_phase: phi div(u).
```

Both residuals include the same inertia and diffusion as the Galerkin terms.
Their position derivatives include moving gradients, Hessians, quadrature
weights, mesh velocity and its divergence. The enlarged psi equation and mesh
forcing law are unchanged; their pre-existing couplings remain active.

The historical fields are differentiated in ALE reference coordinates. This
is a consistent ALE extension, not a discrete geometric conservation theorem.
The closed deforming-domain tests can therefore show a small phase-integral
drift despite a tightly satisfied assembled power identity. AMR transfers can
also change energy or mass; their error is not included as an artificial
physical dissipation.

## Mobilities, correction and stabilization

All existing derivatives of `M`, `grad(M)` and `C` remain in Galerkin and strong
residuals and in the direction `a+J/rho`. In particular, diffusion/Hessians are
needed by the momentum strong residual even when tracer SUPG is disabled.
The formula for tau is unchanged. Tau and the correction's global mobility
maximum are frozen only in the approximate Jacobian and its FD diagnostic;
the residual updates them normally.

The reference energy model refuses stabilization. Here momentum SUPG, PSPG and
tracer SUPG use the modified strong residuals. Their signed powers are evaluated
with energy tests `u`, `-p`, and `mu`, respectively. PSPG thus contributes
`+tau/rho R_momentum.grad(p)` to the balance. They are not assumed dissipative
individually or collectively.

## Sources, boundaries and diagnostic implementation

MMS forces the continuous modified PDE, not a timestep-dependent discrete
identity. The sources include the conservative phase term and skew mass term,
clipped viscosity and its derivative, and the positive-density inertia. Outside
[-1,1], ALE also needs the difference between positive and affine inertia under
mesh motion; the exact position map is inverted to evaluate the exact mesh
velocity in reference coordinates. Smooth overshooting MMS runs test this
extension. Crossing the clipping kink is not a smooth manufactured case.

Unlike the optional legacy source-gradient path, this formulation evaluates
source gradients automatically for ALE MMS. Without them the position columns
of the analytic Jacobian omit how quadrature points move through the source.
They are still centered numerical source gradients; analytic higher derivatives
have not been introduced.

No open-boundary flux integrals were ported: the reference restricts its boundary
conditions too. This adaptation checks every actual boundary ID collectively
in MPI, including unspecified IDs. Supported physical walls are impermeable
and, for ALE, have fixed boundary positions. Interior mesh deformation and full
MMS traces are allowed. Other moving physical walls need further boundary work.

The four postprocessing tables adapt the reference's domain and phase energy
integrals to Abels, adding correction and stabilization powers. They share the
assembly strong-residual functions to avoid a second equation implementation.
The reference's detailed quasi-incompressible CSV columns (including BDF2
modified-energy remainders) are not copied under misleading Abels labels;
physical energies and the assembled signed-power balance are reported here.

Checkpoint sidecars always replace old state, even if monitoring is disabled.
The new formulation rejects stationary integration, as does the reference
energy model. Existing stationary Abels runs may use the diagnostics; their
non-temporal source/transport/stabilization powers are evaluated.

## Verification record

Local Release build with the corrected deal.II and reference BLAS:

- Temporal BDF1 identities, discrete potential chain rule, BDF2 kinetic
  G-identity, and kernel derivatives, including phase overshoots.
- Fixed/ALE/enlarged Jacobians with both, momentum-only, tracer-only and no
  stabilization: maximum absolute errors below `1.7e-7` on the MMS fixtures.
- Other mobility laws (constant, degenerate and adaptive velocity), both
  stabilizations and all three solvers: nine additional Jacobians below `6.4e-7`.
- Enlarged 3D closed-domain Jacobian: `1.34e-7`.
- Existing profile-correction Jacobian fixtures, option disabled: all converge;
  outputs agree with the existing references at the standard deal.II
  comparison tolerances (absolute 1e-6, relative 1e-8).
- A closed fixed case with adaptive normal mobility, profile correction and
  both stabilizations satisfies the signed-power balance within `3.2e-12`.
- BDF1/BDF2 closed fixed domain, stabilization on/off: decreasing physical
  energy in this case; power-balance residual below `3e-12`, phase drift below
  `1e-16`. This is a test result, not a general monotonicity proof.
- Internally deforming closed ALE/enlarged BDF2 cases with a solenoidal initial
  vortex: power residual below `6e-13`; phase drift about `1.05e-9` / `1.78e-9`
  over the tested interval.
- Split versus uninterrupted BDF2 runs at one and two MPI ranks preserve the
  diagnostic reference and give identical final reported total energy.
- Three-level space/time MMS (P2/P1 velocity/pressure, P2 phase/potential,
  adaptive normal velocity mobility, profile correction, both stabilizations):

| Solver | u | p | phi | mu | position | psi |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 3.03 | 2.20 | 2.58 | 2.41 | — | — |
| ALE | 3.02 | 1.99 | 2.40 | 2.19 | 3.04 | — |
| Enlarged ALE | 3.02 | 1.99 | 2.40 | 2.19 | 3.04 | 2.46 |

These are the last observed combined refinement rates, not pure temporal orders.
Two-level overshooting MMS without profile correction also converge in all three
solvers (velocity about 3.06, phase about 3.26, potential about 3.60).

## Retained nonlinear limitation: near-rest SUPG length switch

`near_rest_enlarged.prm` is a deliberately retained reproducer, not a passing CI
test. With BDF2 startup, zero initial velocity and the prescribed internal mesh
forcing, the energy formulation can stall near `t=0.004`. The baseline formulation
and the variant without momentum stabilization converge. Removing tracer SUPG
does not remove the failure; changing the startup fraction to 0.5 does not either.

A temporary, read-only assembly diagnostic found the same quadrature point
crossing the existing `|u-w|^2=1e-12` switch in the 2D streamline length:

| squared speed | momentum tau | global residual norm |
|---:|---:|---:|
| 9.9995932673e-13 | 9.9926353437e-4 | 9.27623899e-8 |
| 1.0000277790e-12 | 9.8310145173e-4 | 1.85972189e-8 |

The switch from the cell diameter to the directional length is discontinuous;
tau changes by about 1.6% across arbitrarily close speeds. The new equations can
place a Newton iterate across this switch even when the old equations did not.
This is a retained limitation of the current SUPG coefficient, not an omitted
analytic derivative discovered by the Jacobian tests. The reference energy
model disables stabilization and therefore does not cover this combination.
No change to tau or Newton is included in this branch. The instrumentation was
removed. A separate change to the near-zero-velocity length transition would
require choosing and validating its mathematical regularization.
