"""Deprecated alias: the product is now DIVERGE (package `diverge`). `import herd.x` returns the
very same module object as `diverge.x`, so old imports and `python -m herd` keep working."""
import importlib
import importlib.abc
import importlib.util
import sys

import diverge as _diverge


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith("herd.") and name != "herd.__main__":
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("diverge" + spec.name[len("herd"):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Alias())
__path__ = _diverge.__path__
