#!/usr/bin/env python3
"""Validate selectable geometric capillarity, Jacobians and matching outputs."""
import argparse
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--ranks', type=int, default=1)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
build = a.build.resolve()
work = build / f'validate-laplace-beltrami-mpi{a.ranks}'
work.mkdir(exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

cases = [(solver, 'jacobian_profile_correction.prm', degree, slip, '')
         for solver in ['incompressible_chns', 'incompressible_chns_ale',
                        'incompressible_chns_ale_enlarged']
         for degree, slip in [(1, False), (2, False), (2, True)]]
cases.append(('incompressible_chns_ale_enlarged', 'jacobian_energy_abels_3d.prm',
              1, False, '_3d'))
for solver, template, degree, slip, suffix in cases:
    base = (root / 'tests' / solver / template).read_text()
    text = re.sub(r'set Tracer degree\s*=\s*2', f'set Tracer degree = {degree}', base)
    text = re.sub(r'set t_end\s*=\s*0.0[23]', 'set t_end = 0.01', text)
    # Smooth nonzero gradient isolates analytic differentiation from the
    # intentionally nonsmooth |g| at g=0.
    text = text.replace('-0.3 + 0.2*x*y', '-0.3 + 0.15*x + 0.2*x*y')
    if slip:
        before, fluid = text.split('subsection Fluid boundary conditions', 1)
        lines = fluid.splitlines(keepends=True)
        depth = 1
        for i, line in enumerate(lines):
            if line.strip().startswith('subsection '):
                depth += 1
            elif line.strip() == 'end':
                depth -= 1
            if depth == 0:
                break
            lines[i] = re.sub(r'set type\s*=\s*input_function', 'set type = slip', line)
        text = before + 'subsection Fluid boundary conditions' + ''.join(lines)
    text += '''
subsection Cahn Hilliard
  set surface tension representation = laplace beltrami
  set enable energy consistent formulation = false
end
subsection Output
  set write vtu results = true
  set output prefix = lb
end
subsection Postprocessing
  subsection surface tension force
    set enable = true
    set write results = true
    set precision = 12
  end
end
'''
    directory = work / (f'{solver}_P{degree}{suffix}' + ('_slip' if slip else ''))
    directory.mkdir(exist_ok=True)
    prm = directory / 'input.prm'
    prm.write_text(text)
    with (directory / 'run.log').open('w') as log:
        subprocess.run(['mpirun', '-n', str(a.ranks), str(build / solver), str(prm)],
                       cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    log = (directory / 'run.log').read_text()
    errors = [float(v) for v in re.findall(
        r'Max absolute error analytical vs fd matrix is ([\deE+.-]+)', log)]
    assert errors and max(errors) < 2e-5, (solver, degree, errors)
    outputs = list(directory.glob('lb_*.pvtu'))
    assert outputs and all('surface_tension_force' in f.read_text() for f in outputs)
    assert all('pressure_abels' not in f.read_text() for f in outputs)
    lines = (directory / 'lb_surface_tension_force.txt').read_text().splitlines()
    assert len(lines) >= 3
    keys = lines[0].split()
    for line in lines[1:]:
        row = dict(zip(keys, map(float, line.split())))
        assert row['integral_magnitude'] + 1e-10 >= row['resultant_norm'], row
    print(f'{solver}{suffix} P{degree} slip={slip}: Jacobian {max(errors):.3g}, TXT and ParaView OK', flush=True)
