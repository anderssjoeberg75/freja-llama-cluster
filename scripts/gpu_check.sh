#!/usr/bin/env bash
# Quick check of GPU state that silently limits inference speed:
# PCIe link (a chipset x4 slot hurts -sm row), clocks, power limit, throttling.
set -euo pipefail

nvidia-smi --query-gpu=index,name,pci.bus_id,persistence_mode,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,power.max_limit,clocks.sm,clocks.mem,clocks.max.mem,temperature.gpu \
  --format=csv
echo
echo "Throttle reasons (should be idle/none outside of load):"
nvidia-smi --query-gpu=index,clocks_event_reasons.active --format=csv 2>/dev/null \
  || nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv
echo
echo "Note: link gen often drops to 1 at idle - run while generating to see the real link."
echo "Persistence mode should be 'Enabled' (nvidia-persistenced)."
