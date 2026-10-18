"""DIVERGE was called HERD: the old package name, CLI and MCP tool names keep working as aliases."""
import subprocess
import sys


def test_herd_package_is_an_alias_of_diverge():
    import diverge.pipeline
    import herd.pipeline
    from herd.contract import NEXT_ACTION
    from diverge import contract
    assert herd.pipeline is diverge.pipeline and NEXT_ACTION is contract.NEXT_ACTION


def test_python_m_herd_still_runs():
    out = subprocess.run([sys.executable, "-m", "herd", "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "diverge" in out.stdout
