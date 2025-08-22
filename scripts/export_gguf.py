#!/usr/bin/env python
"""
GGUF Conversion Stub

Manual steps for GGUF conversion:

1. Clone llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make

2. Convert HF model to GGUF:
   python convert.py path/to/model --outdir ./models --outfile model.gguf

3. Quantize GGUF (optional):
   ./quantize ./models/model.gguf ./models/model-q4_0.gguf q4_0

4. Test with llama.cpp:
   ./main -m ./models/model-q4_0.gguf -p "سلام نړۍ" -n 64

For automation, consider using:
- https://github.com/ggerganov/llama.cpp/blob/master/convert.py
- Or community tools like text-generation-webui conversion scripts

Supported quantization types: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32
"""

if __name__ == "__main__":
    print(__doc__)
