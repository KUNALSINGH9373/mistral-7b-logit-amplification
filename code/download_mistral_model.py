#!/usr/bin/env python3
"""
Download and verify Mistral 7B Instruct model.
Includes options for different quantization levels based on system resources.
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


# Model identifier
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def get_available_memory_gb():
    """Get available system RAM in GB"""
    return torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else psutil.virtual_memory().available / (1024**3)


def download_with_progress(model_id: str):
    """Download model files and show progress"""
    print(f"\n📥 Downloading {model_id}")
    print("=" * 60)

    try:
        # Get list of files
        print("Fetching file list...")
        files = list_repo_files(repo_id=model_id)
        model_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]

        print(f"Found {len(model_files)} model files to download")

        # Download tokenizer first (small, fast)
        print("\n1️⃣  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("   ✅ Tokenizer downloaded")

        # Download model with transformers (handles caching automatically)
        print("\n2️⃣  Downloading model weights...")
        print("   (This may take 10-20 minutes depending on connection)")
        print("   Weights: ~14GB (safetensors format)")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("   ✅ Model downloaded and loaded")

        return tokenizer, model

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


def verify_model(tokenizer, model):
    """Test that model can generate text"""
    print("\n3️⃣  Verifying model functionality...")
    print("-" * 60)

    try:
        # Test prompt
        test_prompt = "What is logit amplification? Answer briefly:"
        print(f"\nTest prompt: '{test_prompt}'")

        # Encode
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nModel response:\n{response}")
        print("\n✅ Model working correctly!")

        return True

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False


def show_model_info(model, tokenizer):
    """Display model statistics"""
    print("\n📊 Model Information:")
    print("-" * 60)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model ID: {MODEL_ID}")
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Model dtype: {model.dtype}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        print(f"Device: CPU")

    cache_path = CACHE_DIR / MODEL_ID.replace("/", "--")
    if cache_path.exists():
        import shutil
        size = sum(f.stat().st_size for f in cache_path.rglob('*')) / (1024**3)
        print(f"Cache size: {size:.2f}GB")


def suggest_quantization():
    """Suggest quantization based on system"""
    print("\n💡 Optimization Suggestions:")
    print("-" * 60)

    try:
        import psutil
        available_ram = psutil.virtual_memory().available / (1024**3)

        if available_ram < 12:
            print("⚠️  Low RAM (<12GB) - Use 4-bit quantization")
            print("   Run with: --quantization 4bit")
        elif available_ram < 20:
            print("⚠️  Medium RAM (12-20GB) - Use 8-bit quantization recommended")
            print("   Run with: --quantization 8bit")
        else:
            print("✅ Sufficient RAM - Can run with float16")
            print("   For even faster inference, use: --quantization 8bit")

    except Exception as e:
        print(f"Could not check RAM: {e}")


def main():
    print("=" * 60)
    print("Mistral 7B Instruct - Model Download & Setup")
    print("=" * 60)

    # Check PyTorch
    print(f"\n🔍 Checking environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download
    print("\n" + "=" * 60)
    tokenizer, model = download_with_progress(MODEL_ID)

    # Verify
    print("\n" + "=" * 60)
    if verify_model(tokenizer, model):
        show_model_info(model, tokenizer)
        suggest_quantization()

        print("\n" + "=" * 60)
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("  1. Review suggestions above")
        print("  2. Run experiments: python logit_diff_mistral.py")
        print("=" * 60)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
