@echo off
echo ============================================================
echo   CS2 Texture Upscaler - Dependency Installer
echo ============================================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/6] Installing PyTorch with CUDA 12.8 (RTX 50-series support)...
pip uninstall torch torchvision -y >nul 2>&1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo Trying cu124...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo Falling back to CPU PyTorch...
        pip install torch torchvision
    )
)

echo.
echo [2/6] Installing build tools...
pip install cython numpy setuptools wheel

echo.
echo [3/6] Patching and installing basicsr (Python 3.13 fix)...
python install_basicsr.py
if errorlevel 1 (
    echo ERROR: basicsr installation failed. See above for details.
    pause
    exit /b 1
)

echo.
echo [4/6] Installing Real-ESRGAN...
pip install realesrgan --no-deps
pip install facexlib gfpgan --no-deps

echo.
echo [5/8] Installing remaining ESRGAN runtime deps...
pip install addict future lmdb pyyaml requests scikit-image tb-nightly tqdm yapf

echo.
echo [6/8] Installing torch.compile acceleration (triton)...
pip install triton-windows 2>nul
if errorlevel 1 (
    echo NOTE: triton-windows not available for this Python version.
    echo       torch.compile will use eager backend instead.
)

echo.
echo [7/8] Installing UI and image dependencies...
pip install Pillow opencv-python PyQt6 imageio

echo.
echo [8/8] Installing image format support...
pip install imageio-ffmpeg

echo.
echo ============================================================
echo   Installation complete! Run: python main.py
echo ============================================================
pause
