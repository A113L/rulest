import builtins
from types import SimpleNamespace

import gpu_diagnostics


def test_missing_pyopencl_returns_actionable_diagnostic(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyopencl":
            raise ModuleNotFoundError("No module named 'pyopencl'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(gpu_diagnostics, "_run_probe", lambda *a, **k: {"available": False, "output": "Not found"})

    report = gpu_diagnostics.collect_gpu_diagnostics()

    assert report["pyopencl_available"] is False
    assert report["gpu_count"] == 0
    assert "not available" in report["summary"].lower()
    assert report["python_executable"]
    assert "-m pip install --upgrade pyopencl" in report["recommended_action"]


def test_opencl_devices_are_enumerated_with_cli_compatible_indices(monkeypatch):
    info = SimpleNamespace(NAME="name", VENDOR="vendor", MAX_COMPUTE_UNITS="units", GLOBAL_MEM_SIZE="memory")
    dtype = SimpleNamespace(GPU=4, CPU=2)

    class Device:
        def __init__(self, name, kind):
            self.values = {"name": name, "vendor": "Test Vendor", "units": 12, "memory": 8 * 1024**3}

        def get_info(self, key):
            return self.values[key]

    gpu = Device("Test GPU", "GPU")
    cpu = Device("Test CPU", "CPU")
    platform = SimpleNamespace(
        name="Test OpenCL Platform", vendor="Test Vendor", version="OpenCL 3.0",
        get_devices=lambda kind: [gpu] if kind == dtype.GPU else [cpu],
    )
    fake_cl = SimpleNamespace(
        VERSION_TEXT="test-version", device_type=dtype, device_info=info,
        get_platforms=lambda: [platform],
    )
    monkeypatch.setitem(__import__("sys").modules, "pyopencl", fake_cl)
    monkeypatch.setattr(gpu_diagnostics, "_run_probe", lambda *a, **k: {"available": False, "output": "Not found"})

    report = gpu_diagnostics.collect_gpu_diagnostics()

    assert report["gpu_count"] == 1
    assert report["cpu_count"] == 1
    assert report["devices"][0]["index"] == 0
    assert report["devices"][0]["name"] == "Test GPU"
    assert report["devices"][0]["type"] == "GPU"
    assert report["devices"][1]["index"] == 1
    assert report["devices"][1]["type"] == "CPU"
    assert "1 GPU" in report["summary"]


def test_opencl_platform_failure_is_explained(monkeypatch):
    fake_cl = SimpleNamespace(
        VERSION_TEXT="test-version",
        device_type=SimpleNamespace(GPU=4, CPU=2),
        device_info=SimpleNamespace(NAME="name", VENDOR="vendor", MAX_COMPUTE_UNITS="units", GLOBAL_MEM_SIZE="memory"),
        get_platforms=lambda: (_ for _ in ()).throw(RuntimeError("No OpenCL platform found")),
    )
    monkeypatch.setitem(__import__("sys").modules, "pyopencl", fake_cl)
    monkeypatch.setattr(gpu_diagnostics, "_run_probe", lambda *a, **k: {"available": False, "output": "Not found"})

    report = gpu_diagnostics.collect_gpu_diagnostics()

    assert report["pyopencl_available"] is True
    assert "No OpenCL platform found" in report["opencl_error"]
    assert "driver" in report["recommended_action"].lower()
