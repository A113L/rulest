#!/usr/bin/env python3
"""Standalone OpenCL/GPU diagnostics used by the GUI and CLI support workflow.

This module intentionally imports PyOpenCL lazily: the web UI must still start
when PyOpenCL or the vendor OpenCL runtime is missing, so it can explain how to
fix the local installation.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _safe_info(device: Any, cl: Any, key: Any, fallback: Any = None) -> Any:
    try:
        return device.get_info(key)
    except Exception:
        return fallback


def _run_probe(command: str, args: list[str], timeout: int = 8) -> dict[str, Any]:
    executable = shutil.which(command)
    if not executable:
        return {"available": False, "path": None, "output": "Not found in PATH."}
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True,
            timeout=timeout, errors="replace", check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = "\n".join(output.strip().splitlines()[:12])[:2400]
        return {
            "available": True,
            "path": executable,
            "return_code": result.returncode,
            "output": output or f"Command exited with code {result.returncode} and no output.",
        }
    except subprocess.TimeoutExpired:
        return {"available": True, "path": executable, "output": f"Timed out after {timeout} seconds."}
    except Exception as exc:  # diagnostics must never prevent the GUI from opening
        return {"available": True, "path": executable, "output": f"Probe failed: {type(exc).__name__}: {exc}"}


def collect_gpu_diagnostics() -> dict[str, Any]:
    """Return structured diagnostics without raising on missing drivers/packages."""
    result: dict[str, Any] = {
        "system": platform.system(),
        "system_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pyopencl_available": False,
        "pyopencl_version": None,
        "pyopencl_error": None,
        "opencl_platforms": [],
        "devices": [],
        "gpu_count": 0,
        "cpu_count": 0,
        "opencl_error": None,
        "probes": {},
        "summary": "GPU detection has not completed.",
        "recommended_action": "Run GPU detection again.",
    }

    try:
        import pyopencl as cl  # type: ignore
        result["pyopencl_available"] = True
        result["pyopencl_version"] = getattr(cl, "VERSION_TEXT", None) or getattr(cl, "__version__", "Installed")
    except Exception as exc:
        result["pyopencl_error"] = f"{type(exc).__name__}: {exc}"

    if result["pyopencl_available"]:
        try:
            platforms = cl.get_platforms()
            for platform_index, plat in enumerate(platforms):
                platform_entry = {
                    "index": platform_index,
                    "name": str(getattr(plat, "name", "Unknown platform")),
                    "vendor": str(getattr(plat, "vendor", "Unknown vendor")),
                    "version": str(getattr(plat, "version", "Unknown version")),
                }
                result["opencl_platforms"].append(platform_entry)

                # Match rulest.devices.get_all_devices() ordering exactly so the
                # displayed index can be passed to --device without remapping.
                for dtype, label in ((cl.device_type.GPU, "GPU"), (cl.device_type.CPU, "CPU")):
                    try:
                        found = plat.get_devices(dtype)
                    except Exception as exc:
                        platform_entry.setdefault("device_query_errors", []).append(
                            f"{label}: {type(exc).__name__}: {exc}"
                        )
                        continue
                    for device in found:
                        name = _safe_info(device, cl, cl.device_info.NAME, getattr(device, "name", "Unknown device"))
                        vendor = _safe_info(device, cl, cl.device_info.VENDOR, getattr(device, "vendor", "Unknown vendor"))
                        units = _safe_info(device, cl, cl.device_info.MAX_COMPUTE_UNITS, None)
                        memory = _safe_info(device, cl, cl.device_info.GLOBAL_MEM_SIZE, None)
                        entry = {
                            "index": len(result["devices"]),
                            "name": str(name),
                            "vendor": str(vendor),
                            "type": label,
                            "platform": platform_entry["name"],
                            "compute_units": int(units) if units is not None else None,
                            "global_memory_bytes": int(memory) if memory is not None else None,
                        }
                        result["devices"].append(entry)
                        result["gpu_count" if label == "GPU" else "cpu_count"] += 1
        except Exception as exc:
            result["opencl_error"] = f"{type(exc).__name__}: {exc}"

    # Extra probes are supporting evidence only; OpenCL enumeration remains the
    # source of truth for devices the engine can actually use.
    system = result["system"]
    if system == "Windows":
        result["probes"]["nvidia-smi"] = _run_probe(
            "nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        result["probes"]["clinfo"] = _run_probe("clinfo", ["--list"])
    elif system == "Linux":
        result["probes"]["nvidia-smi"] = _run_probe(
            "nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        result["probes"]["clinfo"] = _run_probe("clinfo", ["--list"])
        result["probes"]["rocminfo"] = _run_probe("rocminfo", [], timeout=10)
    elif system == "Darwin":
        result["probes"]["system_profiler"] = _run_probe(
            "system_profiler", ["SPDisplaysDataType"]
        )

    if result["gpu_count"]:
        result["summary"] = f"OpenCL ready · {result['gpu_count']} GPU device(s) detected."
        result["recommended_action"] = "No driver action is required. Choose a listed GPU or leave the device field blank for automatic selection."
    elif not result["pyopencl_available"]:
        result["summary"] = "PyOpenCL is not available in the Python environment running this GUI."
        result["recommended_action"] = (
            f'Install PyOpenCL into this exact Python environment: "{sys.executable}" -m pip install --upgrade pyopencl. '
            "Then restart the GUI. If installation succeeds but OpenCL platforms are still missing, install/update the GPU vendor driver/runtime."
        )
    elif result["opencl_error"]:
        result["summary"] = "PyOpenCL loaded, but the OpenCL runtime could not enumerate platforms."
        result["recommended_action"] = (
            "Install or repair the official graphics driver for your GPU (NVIDIA, AMD or Intel), then reboot. "
            "PyOpenCL is only the Python binding; it does not install the GPU's OpenCL driver/runtime."
        )
    elif not result["opencl_platforms"]:
        result["summary"] = "PyOpenCL loaded, but no OpenCL platform is installed."
        result["recommended_action"] = (
            "Install or repair the official graphics driver/OpenCL runtime for your GPU, reboot, and run detection again. "
            "Installing the Python package alone is not enough."
        )
    elif not result["gpu_count"]:
        result["summary"] = f"OpenCL platform(s) found, but no GPU device is exposed ({result['cpu_count']} CPU device(s))."
        result["recommended_action"] = (
            "Update/reinstall the official GPU driver and make sure the GPU is enabled in Device Manager/BIOS. "
            "On laptops, set the terminal/Python app to use the high-performance GPU in the OS graphics settings."
        )

    return result


def main() -> int:
    report = collect_gpu_diagnostics()
    print("RULEST GPU / OPENCL DIAGNOSTICS")
    print(f"System: {report['system']} {report['system_release']} ({report['machine']})")
    print(f"Python: {report['python_version']}")
    print(f"Python executable: {report['python_executable']}")
    print(f"PyOpenCL: {report['pyopencl_version'] or 'NOT AVAILABLE'}")
    if report["pyopencl_error"]:
        print(f"PyOpenCL error: {report['pyopencl_error']}")
    print(f"OpenCL platforms: {len(report['opencl_platforms'])}")
    for item in report["devices"]:
        mem = item["global_memory_bytes"]
        mem_text = f", {mem / (1024 ** 3):.1f} GB" if mem else ""
        print(f"  {item['index']}: {item['name']} ({item['type']}) - {item['vendor']}{mem_text}")
    if report["opencl_error"]:
        print(f"OpenCL error: {report['opencl_error']}")
    print(f"\nSTATUS: {report['summary']}")
    print(f"ACTION: {report['recommended_action']}")
    for name, probe in report["probes"].items():
        if probe.get("available") and probe.get("output"):
            print(f"\n{name}:\n{probe['output']}")
    print("\nFull JSON report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gpu_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
