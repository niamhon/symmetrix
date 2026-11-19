"""Install into a fresh tree, repeat safely, and protect existing source files."""
from pathlib import Path
import subprocess
import pytest

REPO = Path(__file__).resolve().parents[3]

def tree(tmp_path):
    root = tmp_path/'lammps with spaces'
    for folder in ['cmake', 'src/KOKKOS']:
        (root/folder).mkdir(parents=True)
    (root/'cmake/CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\n')
    return root

def install(root):
    return subprocess.run(['bash', str(REPO/'pair_symmetrix/install.sh'), str(root)], text=True, capture_output=True, stdin=subprocess.DEVNULL)

def test_install_repeat_and_lsc_sources(tmp_path):
    root = tree(tmp_path)
    first = install(root)
    assert first.returncode == 0, first.stderr
    text = (root/'cmake/CMakeLists.txt').read_text()
    assert 'CUDA::cudart' in text
    for name in ['pair_symmetrix_mace_lsc.cpp', 'pair_symmetrix_mace_lsc.h', 'pair_symmetrix_mace_lsc_mesh.cpp', 'pair_symmetrix_mace_lsc_mesh.h']:
        assert (root/'src'/name).resolve() == REPO/'pair_symmetrix'/name
    for name in ['pair_symmetrix_mace_lsc_kokkos.cpp', 'pair_symmetrix_mace_lsc_kokkos.h']:
        assert (root/'src/KOKKOS'/name).resolve() == REPO/'pair_symmetrix'/name
    assert install(root).returncode == 0
    assert text == (root/'cmake/CMakeLists.txt').read_text()

@pytest.mark.parametrize('conflict', ['source', 'cmake'])
def test_install_rejects_conflicting_edits(tmp_path, conflict):
    root = tree(tmp_path)
    target = root/('src/pair_symmetrix_mace_lsc.cpp' if conflict == 'source' else 'cmake/CMakeLists.txt')
    original = '// user source\n' if conflict == 'source' else '# SYMMETRIX manual integration\n'
    target.write_text(original)
    assert install(root).returncode != 0
    assert target.read_text() == original
