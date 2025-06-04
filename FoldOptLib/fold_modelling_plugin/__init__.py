# Compatibility wrapper to support legacy import paths used in tests.
# Re-export objects from the main package modules.
# Avoid importing heavy modules at import time. Instead expose them lazily by
# referencing the corresponding modules from the main package when accessed.
from importlib import import_module


def __getattr__(name):
    return import_module(f"FoldOptLib.{name}")

__all__ = []
