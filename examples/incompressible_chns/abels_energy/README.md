# Experimental energy-consistent Abels formulation

Enable the coupled formulation with:

```text
subsection Cahn Hilliard
  set enable energy consistent formulation = true
end
```

The default is `false`. The option couples conservative phase transport, skew
momentum transport and energy-weighted BDF inertia. It supports CHNS, CHNS-ALE
and enlarged CHNS-ALE, the existing mobility models, reconstructed implicit
profile correction and either stabilization switch.

This remains the volume-averaged incompressible Abels model. It is not the
quasi-incompressible model from the reference branch. See [AUDIT.md](AUDIT.md)
for the equations and the differences from that implementation.

## Supported boundaries and time integration

Use BDF1 or BDF2. BDF2 requires `bdf start method = BDF1` and a start-step
ratio strictly between zero and one, as in the reference implementation. Physical boundaries must use fluid `no_slip` or `slip`, together
with phase `no_flux`. For ALE, their mesh position must use `fixed`; the interior
mesh may deform. Full `velocity_mms` / `dirichlet_mms` traces are also supported
for manufactured solutions. Open boundaries and moving physical walls are
rejected: their boundary flux contributions have not been implemented.

For energy-identity checks, use the same degree for phase and potential and a
stable velocity/pressure pair, e.g. P2/P1. Other discretizations do not inherit
an energy theorem merely by selecting this option.

The new formulation overrides `enable tracer limiter` for material properties:
clamped phase is used for kinetic density and viscosity; affine density is used
for transport and gravity. The finite-element phase itself is never clipped.
SUPG/PSPG still uses the existing `J/rho` and `1/rho` operators and therefore
requires positive affine density. The positive kinetic extension does not make
these divisions safe for arbitrary phase overshoots.

## Diagnostics

The following independent sections use the usual postprocessing settings
(`enable`, `write results`, `output frequency`, `precision`, `output prefix`):

```text
subsection Postprocessing
  subsection chns conservation
    set enable = true
    set write results = true
  end
  subsection chns energy
    set enable = true
    set write results = true
  end
  subsection chns phase energy
    set enable = true
    set write results = true
  end
  subsection chns stabilization energy
    set enable = true
    set write results = true
  end
end
```

The tables contain domain volume, phase integral, affine/positive masses,
kinetic/bulk/gradient/gravitational energies, physical dissipations, profile
correction work rate, and separate momentum SUPG, PSPG and tracer SUPG powers.
Phase energies partition quadrature points by the sign of phase; their sum is
the domain energy. Stabilization powers are signed: positive removes energy in
the assembled balance, negative adds it. Their accumulated works use the
accepted-step right-endpoint sum `sum(dt * power)`.

`volume_free_kinetic_power_residual` tests the assembled kinetic-plus-free-energy
volume identity. It includes body-force power, source power, correction power,
transport/geometric power and stabilization; it is not a finite difference of
the reported total energy and excludes boundary work. The gravitational energy
is reported separately. MMS boundary data can do external work, so that volume
residual need not vanish for MMS runs.

Temporal/residual powers unavailable at the initial or restart snapshot are
`nan`, not physical zeros. Reference values, tables and accumulated stabilization
work survive checkpoint/restart in the `.chns_energy` sidecar. Old checkpoints
without that sidecar start a new diagnostic reference.

## Closed example and validation

Run `closed_relaxation.prm` from a dedicated output directory with the fixed-mesh
`incompressible_chns` executable. It uses no external forcing and impermeable
walls. To exercise the table checks and checkpoint continuation automatically:

```bash
python3 tests/incompressible_chns/validate_abels_energy.py \
  --build build-release --ranks 1
```

The validator writes its runs under the specified build directory. It checks
BDF1/BDF2, stabilization on/off, the phase partition and restart continuity. MPI
comparisons may also be run with `--ranks 2`.

BDF1's fixed-mesh algebraic identities are checked by `chns_energy_time`.
BDF2 uses square-root-density inertia and an implicit cubic potential; neither
the source branch nor this adaptation provides an unconditional total-energy
decay theorem for that complete BDF2 scheme. ALE geometry, variable timesteps,
profile correction, stabilization, forcing and mesh transfer likewise require
examining the actual balance rather than assuming monotone physical energy.

The retained `near_rest_enlarged.prm` reproducer documents a nonlinear convergence
limitation at the existing near-zero-speed SUPG length switch. It is not an
example expected to complete successfully; see the measured values in AUDIT.md.
