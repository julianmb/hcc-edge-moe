#!/usr/bin/env bash
set -euo pipefail

echo "=== HCC System Validation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# --- Hardware Check ---
echo ""
echo "--- CPU ---"
grep 'model name' /proc/cpuinfo | head -2
echo "Cores: $(nproc)"

echo ""
echo "--- Memory ---"
free -h

echo ""
echo "--- ROCm / NPU ---"
if command -v rocminfo &>/dev/null; then
    rocminfo 2>/dev/null | grep -E 'Name:|Marketing Name:|Device Type:' | head -10
else
    echo "rocminfo not found"
fi

echo ""
echo "--- XDNA Driver ---"
ls /dev/accel/ 2>/dev/null && echo "accel devices found" || echo "no /dev/accel"
lsmod | grep xdna && echo "xdna driver loaded" || echo "xdna driver not loaded"

echo ""
echo "--- ROCm Version ---"
cat /opt/rocm/.info/version 2>/dev/null || dpkg -l | grep rocm 2>/dev/null | head -3 || echo "ROCm version unknown"

echo ""
echo "--- GPU ---"
lspci 2>/dev/null | grep -i 'vga\|3d\|display\|amd' | head -5 || echo "no lspci"

echo ""
echo "=== Building HCC ==="
cd "$(dirname "$0")"
if command -v rustc &>/dev/null; then
    rustc --version
    cargo build --release 2>&1
    echo "Build: OK"
else
    echo "Rust not installed"
    exit 1
fi

echo ""
echo "=== Running Tests ==="
cargo test 2>&1
echo ""
echo "=== Validation Complete ==="
