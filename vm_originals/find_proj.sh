#!/bin/bash
# List project-like files (excluding system dirs) + home directory layout
find /home /root -maxdepth 6 -type f \( -name "*.py" -o -name "*.html" -o -name "*.md" \
  -o -name "*.ipynb" -o -name "*.csv" -o -name "*.txt" -o -name "*.sh" -o -name "*.json" \) \
  2>/dev/null | grep -viE 'dist-packages|site-packages|node_modules|\.cache|\.local|\.config|\.ssh' \
  > /tmp/proj_files.txt
ls -la /home > /tmp/home_listing.txt 2>/dev/null
echo "proj_files: $(wc -l < /tmp/proj_files.txt) lines"
echo "home listing saved"
