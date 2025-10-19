# Deploying to Hugging Face Space

## Quick Start

Update your existing Hugging Face Space with the latest files:

```bash
cd /workspaces/ZamAI-Phi-3-Mini-Pashto/hf_space

# Login to Hugging Face (first time only)
/bin/python3 -c "from huggingface_hub import login; login()"
# Paste your token from: https://huggingface.co/settings/tokens

# Update your space (replace with your actual Space ID)
/bin/python3 create_space.py tasal9/YOUR-SPACE-NAME
```

## What Gets Uploaded

The script will upload these files to your Space:
- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies
- `README.md` - Space documentation
- `safety/filter.py` - Safety filtering module (if exists)

## Configuration

Before deploying, verify these settings in `app.py`:

```python
MODEL_ID = "tasal9/ZamZeerak-Phi3-Pashto"  # Your model on HF Hub
```

Make sure this matches your actual model repository on Hugging Face.

## After Deployment

1. Visit your Space URL (printed by the script)
2. Wait 2-3 minutes for the Space to rebuild
3. Check the "Logs" tab if there are any build errors
4. Test the interface with sample prompts

## Troubleshooting

**Authentication Error**: Make sure you're logged in with a token that has write access
```bash
/bin/python3 -c "from huggingface_hub import login; login()"
```

**Space Not Found**: Verify your Space ID format is `username/space-name`

**Build Errors**: Check the Logs tab on your Space page for Python errors
