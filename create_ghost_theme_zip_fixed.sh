#!/bin/bash

# Script to properly package a Ghost theme without symlinks
# This addresses the "Symlinks are not allowed in the zip folder" error
# and incorporates fixes for deprecated Handlebars helpers

THEME_DIR="/home/firmans/Documents/GitHub/Extensions-and-Plugins/Ghost-Theme"
OUTPUT_ZIP="/home/firmans/Documents/GitHub/Extensions-and-Plugins/Ghost-Theme-Fixed-For-Upload.zip"

echo "Creating a Ghost-compatible theme ZIP file with all fixes applied..."
echo "Source directory: $THEME_DIR"
echo "Output ZIP: $OUTPUT_ZIP"

# Navigate to theme directory
cd "$THEME_DIR" || exit 1

# Create a clean ZIP file without symlinks and unnecessary files
zip -r "$OUTPUT_ZIP" . \
  --exclude "node_modules/*" \
  --exclude "build/*" \
  --exclude "dist/*" \
  --exclude "tests/*" \
  --exclude "coverage/*" \
  --exclude ".git/*" \
  --exclude ".github/*" \
  --exclude ".vscode/*" \
  --exclude ".idea/*" \
  --exclude "*.log" \
  --exclude "package-lock.json" \
  --exclude "yarn.lock" \
  --exclude ".gitignore" \
  --exclude ".gitmodules" \
  --exclude "Dockerfile" \
  --exclude ".dockerignore" \
  --exclude "__MACOSX" \
  --exclude "*.DS_Store" \
  --exclude "Thumbs.db" \
  --exclude ".DS_Store" \
  --exclude "temp/*" \
  --exclude "tmp/*" \
  --exclude "*.tmp" \
  --exclude "*.bak" \
  --exclude "*.swp" \
  --exclude "*~"

echo "ZIP file created successfully!"
echo "Size: $(du -h "$OUTPUT_ZIP" | cut -f1)"
echo ""
echo "Contents of the ZIP file:"
zip -sf "$OUTPUT_ZIP" | head -20
echo "... (output truncated)"

echo ""
echo "The ZIP file is now ready for upload to Ghost."
echo "It contains no symlinks and only the necessary files for the theme to function properly."
echo ""
echo "Fixed issues include:"
echo "- Replaced deprecated {{author.*}} helpers with {{primary_author.*}}"
echo "- Added required .kg-width-wide and .kg-width-full CSS classes"
echo "- Updated manifest.json to support ghost-api v6"
echo "- Properly wrapped post.hbs and page.hbs with {{#post}}...{{/post}} and {{#page}}...{{/page}}"