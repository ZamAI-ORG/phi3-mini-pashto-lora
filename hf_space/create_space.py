#!/usr/bin/env python3
"""
Script to update/push files to an existing Hugging Face Space for the ZamAI Phi-3 Pashto model.
Run this script to update your existing space on Hugging Face Hub.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def update_space(space_id=None):
    """Update an existing Hugging Face Space with latest files."""
    
    # Configuration - Update this with your existing Space ID
    if space_id is None:
        # Try to read from command line argument
        if len(sys.argv) > 1:
            space_id = sys.argv[1]
        else:
            # Default - update this to your actual Space ID
            username = "tasal9"
            space_name = "ZamZeerak-Phi3-Pashto"  # Update with your actual Space name
            space_id = f"{username}/{space_name}"
    
    print(f"🔄 Updating Space: {space_id}")
    
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
    
    
    try:
        # Initialize API
        api = HfApi()
        
        # Check if space exists
        print("🔍 Checking if Space exists...")
        try:
            _ = api.space_info(repo_id=space_id)
            print(f"✅ Found existing Space: https://huggingface.co/spaces/{space_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify Space exists: {e}")
            print("Continuing with upload anyway...")
        
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
        
        print("\n✅ Space updated successfully!")
        print(f"🌐 Visit your space at: https://huggingface.co/spaces/{space_id}")
        print("\n⏳ Note: It may take a few minutes for the space to rebuild.")
        print("💡 You can check the build logs on the Hugging Face Space page.")
        
    except Exception as e:
        print(f"\n❌ Error updating space: {e}")
        print("\n💡 Make sure you're logged in to Hugging Face:")
        print("   Run: python3 -c 'from huggingface_hub import login; login()'")
        print("   Or set HF_TOKEN environment variable")


if __name__ == "__main__":
    # Usage: python create_space.py [space_id]
    # Example: python create_space.py tasal9/ZamZeerak-Phi3-Pashto
    update_space()
