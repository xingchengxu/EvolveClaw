#!/usr/bin/env bash
# Full search: R(4,4) on n=17 with more generations
set -e
cd "$(dirname "$0")/.."
echo "=== EvolveClaw-Ramsey Search ==="
echo "Running R(4,4) search on n=17"
echo ""
python -m evolveclaw_ramsey.cli run --config configs/demo.yaml
echo ""
echo "Done! Check the runs/ directory for results."
