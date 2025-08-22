#!/usr/bin/env bash
#
# Afghan Ministry of Education PDF Book Downloader
# 
# This script downloads all PDF books from the Afghan MOE curriculum website.
# Usage: ./scripts/download_afghan_books.sh [output_directory] [delay_seconds]
#

set -euo pipefail

# Default values
AFGHAN_MOE_URL="https://moe.gov.af/index.php/ps/%D8%AF-%D9%86%D8%B5%D8%A7%D8%A8-%DA%A9%D8%AA%D8%A7%D8%A8%D9%88%D9%86%D9%87"
OUTPUT_DIR=${1:-"afghan_books"}
DELAY=${2:-"2.0"}

echo "========================================"
echo "Afghan MOE PDF Book Downloader"
echo "========================================"
echo "Source URL: $AFGHAN_MOE_URL"
echo "Output Directory: $OUTPUT_DIR"
echo "Delay between downloads: ${DELAY}s"
echo "========================================"
echo ""

echo "Legal Notice:"
echo "This script is intended for educational and research purposes only."
echo "Please ensure you have permission to download content and comply with"
echo "the website's terms of service and copyright laws."
echo ""

read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo "Starting download..."
python scripts/download_pdf_books.py \
    --url "$AFGHAN_MOE_URL" \
    --output-dir "$OUTPUT_DIR" \
    --delay "$DELAY" \
    --verbose

echo ""
echo "Download completed! Check the '$OUTPUT_DIR' directory for downloaded books."