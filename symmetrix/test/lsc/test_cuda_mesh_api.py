"""The optional CUDA mesh interface also links in CPU-only builds."""
import subprocess
import pytest
import symmetrix
from .reference import REPO

pytestmark=pytest.mark.lsc_reference


def test_cuda_mesh_build_flag():
    assert symmetrix.lsc_cuda_mesh_compiled()==symmetrix.lsc_cuda_compiled()
    if not symmetrix.lsc_cuda_mesh_compiled():
        with pytest.raises(RuntimeError,match='not built'):symmetrix.LSCCudaMesh()


def test_optional_cuda_mesh_stub(tmp_path):
    source=tmp_path/'stub.cpp';source.write_text('#include "mace_lsc_cuda_mesh.hpp"\n#include <stdexcept>\nint main(){if(lsc_cuda_mesh_compiled())return 1;try{LSCCudaMesh mesh;}catch(const std::runtime_error&){return 0;}return 2;}\n')
    binary=tmp_path/'stub'
    result=subprocess.run(['g++','-std=c++20','-I'+str(REPO/'libsymmetrix/source'),str(source),str(REPO/'libsymmetrix/source/mace_lsc_cuda_mesh_stub.cpp'),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    subprocess.run([str(binary)],check=True)
