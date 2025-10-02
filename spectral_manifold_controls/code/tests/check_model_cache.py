"""
Quick check for HuggingFace model cache location and contents.
"""

import os
from pathlib import Path

print("="*80)
print("HuggingFace Model Cache Check")
print("="*80)

# Check cache location
cache_dir = os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE') or Path.home() / '.cache' / 'huggingface'
cache_path = Path(cache_dir)

print(f"\nCache directory: {cache_path}")
print(f"Exists: {cache_path.exists()}")

if cache_path.exists():
    # Check for gpt2-medium
    hub_cache = cache_path / 'hub'
    if hub_cache.exists():
        models = list(hub_cache.glob('models--*gpt2*'))
        print(f"\nGPT-2 models found: {len(models)}")
        for model in models:
            print(f"  - {model.name}")
            # Check size
            total_size = sum(f.stat().st_size for f in model.rglob('*') if f.is_file())
            print(f"    Size: {total_size / (1024*1024):.1f} MB")
    else:
        print("\n[WARNING] Hub cache not found - models may not be cached!")
else:
    print("\n[WARNING] Cache directory doesn't exist!")
    print("Models will be downloaded on first run.")

print("\n" + "="*80)
print("To set custom cache location:")
print("  export HF_HOME=/path/to/cache")
print("  or set TRANSFORMERS_CACHE=/path/to/cache")
print("="*80)

