#!/bin/bash
# 运行全部测试
set -e
echo "=== Running all tests ==="
cd "$(dirname "$0")/.."
pytest test/ -v
echo "=== Done ==="
