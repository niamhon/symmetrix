import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "lsc_organic: opt-in frozen benzene/EMC native-reference pilot")
    config.addinivalue_line("markers", "lsc_lammps: requires the LSC LAMMPS development build")
    config.addinivalue_line(
        "markers", "lsc_cuda: requires the CUDA build and an allocated GPU"
    )
    config.addinivalue_line(
        "markers", "lsc_reference: requires the frozen stage-2 LSC reference checkpoint"
    )


@pytest.fixture(scope="module", autouse=True)
def reference_precision():
    import torch

    previous_dtype = torch.get_default_dtype()
    previous_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    yield
    torch.set_default_dtype(previous_dtype)
    torch.set_num_threads(previous_threads)


@pytest.fixture(scope="module")
def lsc_checkpoint():
    from .reference import MODEL_SHA256, model_path, sha256

    path = model_path()
    assert path.is_file(), f"Missing frozen LSC checkpoint: {path}"
    assert sha256(path) == MODEL_SHA256, "The frozen checkpoint has changed"
    return path


@pytest.fixture(scope="module")
def lsc_model(lsc_checkpoint):
    from .reference import make_calculator

    # This is the original ASE calculator; no geometry evaluation is needed.
    return make_calculator(lsc_checkpoint, "realspace").model
