#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

echo "============================================================"
echo "1. Executing Multi-Database ECG Rhythm Classification Benchmark"
echo "============================================================"
python3 scripts/execute_benchmark.py

echo ""
echo "============================================================"
echo "2. Running Automated Unit & Integration Test Suites"
echo "============================================================"
pytest tests/test_data_pipeline.py tests/test_method_a.py tests/test_method_b.py tests/test_method_c.py tests/test_benchmark_suite.py -v

echo ""
echo "============================================================"
echo "ALL BENCHMARKS AND VERIFICATION SUITES COMPLETED SUCCESSFULLY"
echo "============================================================"
