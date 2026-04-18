"""
Compatibility patches for basicsr on newer torchvision / Python versions.

Must be imported BEFORE basicsr.
"""

import importlib
import sys


def apply_patches():
    """Apply all needed compatibility patches."""
    _patch_torchvision_functional_tensor()
    _patch_basicsr_version()


def _patch_torchvision_functional_tensor():
    """
    basicsr imports torchvision.transforms.functional_tensor which was
    removed in torchvision >= 0.18. Create a shim module that redirects
    to the new location.
    """
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return  # Already available

    try:
        importlib.import_module(module_name)
        return  # Module exists natively
    except ModuleNotFoundError:
        pass

    # Build a shim from torchvision.transforms.functional
    import types
    import torchvision.transforms.functional as F

    shim = types.ModuleType(module_name)
    shim.__package__ = "torchvision.transforms"

    # Map the functions basicsr actually uses
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    # Add any other functions that might be referenced
    for attr in dir(F):
        if not hasattr(shim, attr):
            try:
                setattr(shim, attr, getattr(F, attr))
            except Exception:
                pass

    sys.modules[module_name] = shim


def _patch_basicsr_version():
    """
    Ensure basicsr.version has __gitsha__, which is expected by
    basicsr.__init__ but may be missing from pip-installed builds.
    Patches the module directly without triggering basicsr.__init__.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("basicsr.version")
        if spec is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, '__gitsha__'):
            mod.__gitsha__ = 'unknown'
        sys.modules["basicsr.version"] = mod
    except Exception:
        pass
