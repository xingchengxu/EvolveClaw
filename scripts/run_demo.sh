#!/usr/bin/env bash
# Quick demo: R(4,4) on n=17
set -e
cd "$(dirname "$0")/.."
echo "=== EvolveClaw-Ramsey Demo ==="
echo "Running R(4,4) search on n=17"
echo ""
python -m evolveclaw_ramsey.cli run --config configs/demo.yaml
echo ""
echo "Done! Check the runs/ directory for results."
