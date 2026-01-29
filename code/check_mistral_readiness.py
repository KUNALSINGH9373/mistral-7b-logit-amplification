#!/usr/bin/env python3
"""
Check system readiness for Mistral 7B Instruct experiment.
Verifies dependencies, RAM, disk space, and GPU availability.
"""

import os
import sys
import psutil
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} - Need 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'transformers', 'accelerate', 'numpy']
    optional = ['bitsandbytes', 'peft']

    print("\n📦 Checking Dependencies:")

    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - REQUIRED")
            all_ok = False

    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ⚠️  {pkg} - optional (recommended for quantization)")

    return all_ok


def check_system_resources():
    """Check RAM and disk space"""
    print("\n💾 System Resources:")

    # RAM check
    ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"  Memory: {ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available")

    if available_ram_gb < 8:
        print(f"  ⚠️  Warning: Only {available_ram_gb:.1f} GB available (need at least 8GB)")
    elif available_ram_gb < 16:
        print(f"  ✅ Sufficient (will need 4-bit or 8-bit quantization)")
    else:
        print(f"  ✅ Excellent (can run with full precision)")

    # Disk check
    disk_usage = shutil.disk_usage("/")
    free_gb = disk_usage.free / (1024**3)
    print(f"  Disk: {free_gb:.1f} GB free")

    if free_gb < 20:
        print(f"  ❌ Warning: Only {free_gb:.1f} GB free (need at least 20GB for model)")
        return False
    else:
        print(f"  ✅ Sufficient")

    return available_ram_gb >= 8 and free_gb >= 20


def check_gpu():
    """Check if GPU is available"""
    print("\n🎮 GPU Status:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available")
            print(f"     Device: {torch.cuda.get_device_name(0)}")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"  ℹ️  No GPU (will use CPU - slower but possible)")
    except Exception as e:
        print(f"  ℹ️  Could not check GPU: {e}")


def check_model_cache():
    """Check Hugging Face cache directory"""
    print("\n🤗 Hugging Face Cache:")
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        size_gb = sum(f.stat().st_size for f in cache_dir.rglob('*')) / (1024**3)
        print(f"  ✅ Cache exists: {size_gb:.2f} GB used")
    else:
        print(f"  ℹ️  No cache yet (will be created on first download)")


def main():
    print("=" * 60)
    print("Mistral 7B Amplification - System Readiness Check")
    print("=" * 60)

    python_ok = check_python_version()
    deps_ok = check_dependencies()
    resources_ok = check_system_resources()
    check_gpu()
    check_model_cache()

    print("\n" + "=" * 60)
    if python_ok and resources_ok:
        print("✅ System is READY for Mistral 7B experiment!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run: python download_mistral_model.py")
        return 0
    else:
        print("❌ System needs attention before proceeding")
        if not python_ok:
            print("  - Upgrade Python to 3.8+")
        if not deps_ok:
            print("  - Run: pip install -r requirements.txt")
        if not resources_ok:
            print("  - Free up disk space or check RAM availability")
        return 1


if __name__ == "__main__":
    sys.exit(main())
