#!/usr/bin/env python3
"""Check standard CHNS volume power balances and diagnostic checkpoint continuity."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import tempfile


def read_table(path):
    lines = path.read_text().splitlines()
    keys = lines[0].split()
    return [dict(zip(keys, map(float, line.split())))
            for line in lines[1:] if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    build = args.build.resolve()
    root = Path(__file__).resolve().parents[2]
    workspace = Path(tempfile.mkdtemp(prefix='validate-chns-diagnostics-', dir=build))
    base = (root / 'examples/incompressible_chns/diagnostics/closed_relaxation.prm').read_text()
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')

    def run(name, parameters):
        directory = workspace / name
        directory.mkdir(exist_ok=True)
        prm = directory / 'input.prm'
        prm.write_text(parameters)
        with (directory / 'run.log').open('w') as log:
            subprocess.run(['mpirun', '-n', str(args.ranks),
                            str(build / 'incompressible_chns'), str(prm)],
                           cwd=directory, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=300)
        return directory

    for order in ['BDF1', 'BDF2']:
        for stabilized in [False, True]:
            text = base.replace('set scheme = BDF1', 'set scheme = ' + order)
            if stabilized:
                text = text.replace('set enable supg = false', 'set enable supg = true')
                text = text.replace('set enable tracer supg = false', 'set enable tracer supg = true')
            name = order + ('_stabilized' if stabilized else '_galerkin')
            directory = run(name, text)
            energy = read_table(directory / 'chns_energy.txt')
            phases = read_table(directory / 'chns_phase_energy.txt')
            conservation = read_table(directory / 'chns_conservation.txt')
            assert len(energy) >= 3, 'The simulation did not advance'
            assert math.isnan(energy[0]['temporal_power'])
            assert all(math.isfinite(row['total_energy']) for row in energy), name
            assert max(abs(row['volume_free_kinetic_power_residual'])
                       for row in energy[1:]) < 1.e-8, name
            for row in conservation:
                assert abs(row['domain_volume'] - 1.) < 1.e-12
                assert abs(row['mass_affine'] - 1.5 - 0.5 * row['phase_integral']) < 1.e-11
            assert len(phases) == len(energy)
            for row, phase in zip(energy, phases):
                assert abs(row['total_energy'] - phase['total_energy_phase_0']
                           - phase['total_energy_phase_1']) < 1.e-11
            if order == 'BDF2' and stabilized:
                reference = energy
                restart_text = text
            print(name + ': energy, phase and power checks passed')

    checkpoint = ('\nsubsection Checkpoint Restart\n'
                  ' set enable checkpoint = true\n'
                  ' set checkpoint frequency = 1\nend\n')
    run('restart', restart_text.replace('set t_end = 0.006', 'set t_end = 0.003')
        + checkpoint)
    directory = run('restart', restart_text + checkpoint.replace(
        'set enable checkpoint = true',
        'set enable checkpoint = true\n set restart = true'))
    restarted = read_table(directory / 'chns_energy.txt')
    assert len(restarted) == len(reference)
    for key in ['reference_time', 'total_energy', 'energy_change_from_reference']:
        assert abs(restarted[-1][key] - reference[-1][key]) < 1.e-10, key
    print('Checkpoint continuation passed')


if __name__ == '__main__':
    main()
