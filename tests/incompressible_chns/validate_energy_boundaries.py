#!/usr/bin/env python3
"""Validate CHNS boundary flux Jacobians, pressure work and MPI diagnostics."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess


def replace_section(text, name, replacement):
    lines = text.splitlines(True)
    start, depth = None, 0
    for i, line in enumerate(lines):
        line = line.strip()
        if start is None:
            if line == 'subsection ' + name:
                start, depth = i, 1
        elif line.startswith('subsection '):
            depth += 1
        elif line == 'end':
            depth -= 1
            if depth == 0:
                return ''.join(lines[:start]) + replacement + '\n' + ''.join(lines[i+1:])
    raise ValueError('Missing section: ' + name)


def read_table(path):
    lines = path.read_text().splitlines()
    keys = lines[0].split()
    return [dict(zip(keys, map(float, row.split()))) for row in lines[1:] if row.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    workspace = build / ('validate-energy-boundaries-mpi' + str(args.ranks))
    workspace.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    base = (root / 'tests/incompressible_chns_ale_enlarged/energy_boundary_flux.prm').read_text()

    def run(name, executable, text, expected_error=None):
        directory = workspace / name
        directory.mkdir(exist_ok=True)
        prm = directory / 'input.prm'
        prm.write_text(text)
        with (directory / 'run.log').open('w') as log:
            result = subprocess.run(
                ['mpirun', '-n', str(args.ranks), str(build / executable), str(prm)],
                cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=600)
        log = (directory / 'run.log').read_text()
        if expected_error:
            assert result.returncode != 0 and expected_error in log, name
            print(name + ': unsupported boundary rejected', flush=True)
            return directory
        assert result.returncode == 0, str(directory / 'run.log')
        errors = [float(x) for x in re.findall(
            r'Max absolute error analytical vs fd matrix is ([^\s]+)', log)]
        assert len(errors) >= 2 and max(errors) < 2.e-6, (name, errors)
        print(name + ': max Jacobian error = ' + str(max(errors)), flush=True)
        return directory

    for suffix in ['', '_ale', '_ale_enlarged']:
        executable = 'incompressible_chns' + suffix
        directory = run(executable, executable, base)
        rows = read_table(directory / 'chns_energy.txt')
        assert len(rows) == 4, (executable, len(rows))  # initial + split startup + BDF2
        assert all(math.isfinite(row['free_kinetic_power_residual']) and
                   abs(row['free_kinetic_power_residual']) < 2.e-8 for row in rows[1:])
        assert any(abs(row['outward_boundary_phase_power']) > 1.e-7 for row in rows[1:])

    # Constant velocity and phase with source (-1,0,0): exact p=x on the unit
    # square. Viscous work is zero, and integral -p u.n is exactly -0.05.
    pressure = replace_section(base, 'Initial conditions', '''subsection Initial conditions
  subsection velocity
    set Function expression = 0.05; 0
  end
  subsection cahn hilliard tracer
    set Function expression = 0.2
  end
end''')
    pressure = pressure.replace('set constrain_v = false', 'set constrain_v = true')
    pressure = pressure.replace('0.15+0.05*cos(t)', '0.05')
    pressure = pressure.replace('set body force = 0, -9.81', 'set body force = 0, 0')
    pressure = pressure.replace('set enable supg        = true', 'set enable supg        = false')
    pressure = pressure.replace('set enable tracer supg = true', 'set enable tracer supg = false')
    pressure += '''\nsubsection Source terms
  subsection fluid
    set Function expression = -1; 0; 0
  end
end\n'''
    directory = run('pressure_work', 'incompressible_chns', pressure)
    for row in read_table(directory / 'chns_energy.txt')[1:]:
        assert abs(row['boundary_traction_power'] + 0.05) < 1.e-10
        assert abs(row['boundary_reaction_power'] + 0.05) < 1.e-10
        assert abs(row['free_kinetic_power_residual']) < 1.e-10
    print('Exact pressure work and discrete reaction work passed', flush=True)

    three_d = (root / 'tests/incompressible_chns_ale_enlarged/jacobian_energy_abels_3d.prm').read_text()
    three_d = three_d.replace('set type = no_slip', '''set type = input_function
    subsection u
      set Function expression = 0.05*cos(t)
    end
    subsection v
      set Function expression = 0
    end
    subsection w
      set Function expression = 0
    end''').replace('set type = fixed', '''set type = input_function
    subsection x
      set Function expression = x+0.05*sin(t)
    end
    subsection y
      set Function expression = y
    end
    subsection z
      set Function expression = z
    end''')
    three_d = three_d.replace('set name = x_max\n    set type = input_function',
                             'set name = x_max\n    set constrain_v = false\n    set type = input_function', 1)
    run('moving_3d', 'incompressible_chns_ale_enlarged', three_d)
    phase_bcs = 'subsection CahnHilliard boundary conditions\n  set number = 4\n'
    for boundary, name in enumerate(['x_min', 'x_max', 'y_min', 'y_max']):
        kind = 'input_function' if boundary == 0 else 'no_flux'
        phase_bcs += (f'  subsection boundary {boundary}\n    set id = {boundary}\n'
                      f'    set name = {name}\n    set type = {kind}\n  end\n')
    unsupported = replace_section(base, 'CahnHilliard boundary conditions',
                                  phase_bcs + 'end\n')
    run('unsupported_phase', 'incompressible_chns', unsupported,
        expected_error='Unsupported boundary id')
    print('All boundary checks passed', flush=True)


if __name__ == '__main__':
    main()
