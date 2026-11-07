
#!/usr/bin/env bash
#
# build_explorer.sh - export my GWAS explorer as a static website.
#
# I wrote the explorer (explorer/) with Shiny for Python and export it with
# Shinylive, so it runs entirely in the browser and can be hosted for free on
# GitHub Pages. No server is needed.
#
# Usage: ./build_explorer.sh [output_folder] (default: docs)
# Needs: pip install shinylive
# Preview: python -m http.server --directory docs 8008 then open localhost:8008
# Publish: commit the output folder and point GitHub Pages at it.
#
# Shinylive only packages files inside the app folder, but the
# explorer imports my gwas/ package. So I assemble explorer/ + gwas/ in a
# temporary folder and export from there. My project files are never modified
# and the analysis code exists in one place only.

set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-docs}"
if ! command -v shinylive > /dev/null; then
echo "shinylive is not installed. Run: pip install shinylive" >&2
exit 1
fi
if [ ! -f explorer/app.py ] || [ ! -d gwas ]; then
echo "Run this from the project root (the folder with explorer/ and gwas/)." >&2
exit 1
fi
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/app/gwas"
cp explorer/*.py explorer/requirements.txt "$BUILD/app/"
cp gwas/*.py "$BUILD/app/gwas/"
shinylive export "$BUILD/app" "$OUTecho "Done. Site written to: $OUT