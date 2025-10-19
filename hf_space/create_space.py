#!/usr/bin/env python3
"""
Script to create and push a Hugging Face Space for the ZamAI Phi-3 Pashto model.
Run this script to deploy the space to Hugging Face Hub.
"""

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def create_space():
    """Create and push the Hugging Face Space."""
    
    # Configuration
    space_name = "ZamAI-Phi3-Pashto-Demo"
    username = "tasal9"  # Update with your HF username
    space_id = f"{username}/{space_name}"
    
    # Files to upload
    space_dir = Path(__file__).parent
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "README.md",
    ]
    
    # Check if files exist
    missing_files = [f for f in files_to_upload if not (space_dir / f).exists()]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return
    
    print(f"🚀 Creating Space: {space_id}")
    
    try:
        # Initialize API
        api = HfApi()
        
        # Create the space repository
        print("📦 Creating Space repository...")
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"✅ Space repository created: https://huggingface.co/spaces/{space_id}")
        
        # Upload files
        print("\n📤 Uploading files...")
        for file in files_to_upload:
            file_path = space_dir / file
            print(f"  - Uploading {file}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file,
                repo_id=space_id,
                repo_type="space",
            )
        
        # Copy safety module if it exists
        safety_path = space_dir.parent / "safety" / "filter.py"
        if safety_path.exists():
            print("  - Uploading safety/filter.py...")
            api.upload_file(
                path_or_fileobj=str(safety_path),
                path_in_repo="safety/filter.py",
                repo_id=space_id,
                repo_type="space",
            )
            
            # Create __init__.py for safety module
            init_content = '"""Safety filtering module."""\n'
            init_path = space_dir / "safety_init.py"
            init_path.write_text(init_content)
            api.upload_file(
                path_or_fileobj=str(init_path),
                path_in_repo="safety/__init__.py",
                repo_id=space_id,
                repo_type="space",
            )
            init_path.unlink()  # Clean up temp file
        
        print("\n✅ Space created successfully!")
        print(f"🌐 Visit your space at: https://huggingface.co/spaces/{space_id}")
        print("\n⏳ Note: It may take a few minutes for the space to build and start.")
        print("💡 You can check the build logs on the Hugging Face Space page.")
        
    except Exception as e:
        print(f"\n❌ Error creating space: {e}")
        print("\n💡 Make sure you're logged in to Hugging Face:")
        print("   Run: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")


if __name__ == "__main__":
    create_space()
