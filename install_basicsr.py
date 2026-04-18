"""
Helper script to download, patch, and install basicsr on Python 3.13+.

basicsr 1.4.2 has a broken setup.py that uses exec(code, locals()) to
extract __version__. Python 3.13 changed locals() semantics so variables
written inside exec() are no longer visible. This script replaces the
entire setup.py with a clean minimal version.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BASICSR_VERSION = "1.4.2"


def download(url: str, dest: Path):
    print(f"  Downloading {url} ...")
    import urllib.request
    urllib.request.urlretrieve(url, str(dest))


def write_fixed_setup(src_dir: Path):
    """Replace setup.py with a minimal working version."""
    setup_py = src_dir / "setup.py"

    # Read original to grab dependencies
    original = setup_py.read_text(encoding="utf-8") if setup_py.exists() else ""

    # Extract install_requires from original if possible
    install_requires = [
        "addict",
        "future",
        "lmdb",
        "numpy",
        "opencv-python",
        "Pillow",
        "pyyaml",
        "requests",
        "scikit-image",
        "scipy",
        "tb-nightly",
        "torch>=1.7",
        "torchvision",
        "tqdm",
        "yapf",
    ]

    new_setup = f'''
from setuptools import find_packages, setup

setup(
    name="basicsr",
    version="{BASICSR_VERSION}",
    description="Open Source Image and Video Restoration Toolbox (patched for Py3.13)",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires={install_requires!r},
)
'''
    setup_py.write_text(new_setup, encoding="utf-8")

    # Also fix basicsr/version.py if it exists
    version_py = src_dir / "basicsr" / "version.py"
    if version_py.exists():
        version_py.write_text(
            f'__version__ = "{BASICSR_VERSION}"\n__gitsha__ = "unknown"\n',
            encoding="utf-8",
        )

    # Fix __init__.py to not use the broken version import
    init_py = src_dir / "basicsr" / "__init__.py"
    if init_py.exists():
        text = init_py.read_text(encoding="utf-8")
        # Replace any dynamic version reading with a simple assignment
        if "__version__" not in text:
            text = f'__version__ = "{BASICSR_VERSION}"\n' + text
            init_py.write_text(text, encoding="utf-8")

    # Remove pyproject.toml to avoid it overriding our setup.py
    pyproject = src_dir / "pyproject.toml"
    if pyproject.exists():
        pyproject.unlink()

    print("  Replaced setup.py with clean minimal version")


def main():
    print(f"Installing basicsr {BASICSR_VERSION} with Python 3.13 patch...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tarball = tmpdir / f"basicsr-{BASICSR_VERSION}.tar.gz"

        # Download
        download(
            f"https://files.pythonhosted.org/packages/source/b/basicsr/basicsr-{BASICSR_VERSION}.tar.gz",
            tarball,
        )

        # Extract
        print("  Extracting...")
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(tmpdir, filter="data")

        src_dir = tmpdir / f"basicsr-{BASICSR_VERSION}"
        if not src_dir.exists():
            dirs = [d for d in tmpdir.iterdir() if d.is_dir()]
            if dirs:
                src_dir = dirs[0]
            else:
                print("ERROR: Could not find extracted source directory")
                return 1

        # Replace setup.py entirely
        write_fixed_setup(src_dir)

        # Install without build isolation so it uses our torch
        print("  Installing patched basicsr...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(src_dir),
             "--no-build-isolation", "--no-deps"],
        )

        if result.returncode != 0:
            print("ERROR: pip install failed for patched basicsr")
            return 1

    print("  basicsr installed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
