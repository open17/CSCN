from __future__ import annotations

from importlib import import_module

__all__ = [
    'CSCN',
    'RunConfig',
    'PreprocessConfig',
    'RunResult',
    'RunSummary',
    'PreprocessResult',
    'run_module',
    'run_directory',
    'build_modules',
    'build_ckm',
]

_EXPORTS = {
    'CSCN': ('cscn.core', 'CSCN'),
    'RunConfig': ('cscn.config', 'RunConfig'),
    'PreprocessConfig': ('cscn.config', 'PreprocessConfig'),
    'RunResult': ('cscn.runner', 'RunResult'),
    'RunSummary': ('cscn.runner', 'RunSummary'),
    'PreprocessResult': ('cscn.preprocess', 'PreprocessResult'),
    'run_module': ('cscn.runner', 'run_module'),
    'run_directory': ('cscn.runner', 'run_directory'),
    'build_modules': ('cscn.preprocess', 'build_modules'),
    'build_ckm': ('cscn.ckm', 'build_ckm'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
