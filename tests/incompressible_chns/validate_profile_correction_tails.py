#!/usr/bin/env python3
"""Check cutoff-free profile correction in fixed/ALE/enlarged CHNS tails.

Run with the reference BLAS used for finite-difference comparisons, e.g.
LD_PRELOAD=/tmp/fez-ubuntu-blas/libblas.so.3 FLEXIBLAS=NETLIB python3
tests/incompressible_chns/validate_profile_correction_tails.py --build build-release
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    work = Path(tempfile.mkdtemp(prefix='fez-profile-tails-'))
    print(f'Results: {work}', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Use Korteweg here to isolate profile derivatives: normalized LB normals
    # are very sensitive to the fixed forward-difference step in almost flat tails.
    # validate_laplace_beltrami.py covers the coupled geometric formulation.
    for solver in ('incompressible_chns', 'incompressible_chns_ale',
                   'incompressible_chns_ale_enlarged'):
        for sign in (-1, 1):
            for projection in (False, True):
                directory = work / f'{solver}_{sign}_{projection}'
                directory.mkdir()
                text = (root / 'tests' / solver /
                        'jacobian_profile_correction.prm').read_text()
                text = re.sub(r'set t_end\s*=\s*0.03', 'set t_end = 0.01', text)
                text = text.replace('-0.3 + 0.2*x*y',
                                    f'{sign}*(0.9999 - 0.00002*x + 0.00001*x*y)')
                text += f'''
subsection Cahn Hilliard
  set surface tension representation = korteweg
  set enable flux correction = {str(projection).lower()}
  set profile correction strength = 5
  set surface tension = 0.072
  set interface thickness = 0.004
  subsection mobility
    set model = constant
    set constant value = 1e-6
  end
end
subsection Output
  set output directory = {directory}/
  set output prefix = tails
end
'''
                prm = directory / 'input.prm'
                prm.write_text(text)
                with (directory / 'run.log').open('w') as log:
                    subprocess.run(
                        ['mpirun', '-n', str(args.ranks), str(build / solver), str(prm)],
                        cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True, timeout=300)
                log = (directory / 'run.log').read_text()
                errors = [float(v) for v in re.findall(
                    r'Max absolute error analytical vs fd matrix is ([\deE+.-]+)', log)]
                assert errors and max(errors) < 2e-5, (directory, errors)
                print(f'{solver} sign={sign} projection={projection}: '
                      f'Jacobian {max(errors):.3g}, tail Newton solve OK',
                      flush=True)


if __name__ == '__main__':
    main()
