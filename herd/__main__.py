"""`python -m herd` is a deprecated alias of `python -m diverge`."""
import runpy

runpy.run_module("diverge", run_name="__main__", alter_sys=True)
