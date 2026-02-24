#!/bin/bash
# serve-html.sh - Quick HTTP server for INDEX.html

set -e

PORT=${1:-8080}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo "🚀 Amplification Barometer HTTP Server"
echo "=================================="
echo ""
echo "📍 Repository: $REPO_ROOT"
echo "🌐 Serving on: http://localhost:$PORT"
echo "📄 File: INDEX.html"
echo ""
echo "✅ Starting Python HTTP server..."
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$REPO_ROOT"

# Python 3
if command -v python3 &> /dev/null; then
    python3 -m http.server $PORT
# Python 2 (fallback)
elif command -v python &> /dev/null; then
    python -m SimpleHTTPServer $PORT
else
    echo "❌ Error: Python not found"
    exit 1
fi
