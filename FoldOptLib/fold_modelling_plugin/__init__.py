"""Compatibility layer exposing FoldOptLib modules under the
``FoldModellingPlugin`` namespace used by the tests."""
from importlib import import_module
import sys

_modules = {
    'fold_modelling': 'FoldOptLib.fold_modelling',
    'helper': 'FoldOptLib.helper',
    'optimisers': 'FoldOptLib.optimisers',
    'objective_functions': 'FoldOptLib.objective_functions',
    'splot': 'FoldOptLib.splot',
    'input': 'FoldOptLib.input',
    'from_loopstructural': 'FoldOptLib.from_loopstructural',
}

for name, target in _modules.items():
    module = import_module(target)
    setattr(sys.modules[__name__], name, module)
    sys.modules[f'{__name__}.{name}'] = module

