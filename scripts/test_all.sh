#!/bin/bash
# 运行全部测试
set -e
echo "=== Running all tests ==="
pytest tests/ -v
echo "=== Done ==="
