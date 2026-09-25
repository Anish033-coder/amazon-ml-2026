#!/bin/zsh
# Laptop safety guard for long runs: stops the pipeline before macOS comes under real strain.
#   tools/watchdog.sh <pid> [max_swap_growth_mb=6000] [min_mem_level_pct=8] [min_free_disk_gb=5]
# Swap is measured relative to its value when the run started (macOS frees swap lazily, so the
# absolute figure can stay high after earlier jobs); memory level is the kernel's own
# "memory available" percentage.
pid=$1
max_growth=${2:-6000}
min_level=${3:-8}
min_disk=${4:-5}
log=/Users/anishkumawat/amazon-ml-2026/experiments/watchdog.log
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1 }
base=$(swap_mb)
echo "$(date '+%H:%M:%S') watching pid $pid (swap baseline ${base}MB)" >> $log
while kill -0 $pid 2>/dev/null; do
  growth=$(( $(swap_mb) - base ))
  level=$(sysctl -n kern.memorystatus_level)
  disk=$(df -g ~ | tail -1 | awk '{print $4}')
  if [ "$growth" -gt "$max_growth" ] || [ "$level" -lt "$min_level" ] || [ "$disk" -lt "$min_disk" ]; then
    echo "$(date '+%H:%M:%S') STOPPED pid $pid: swap_growth=${growth}MB mem_level=${level}% free_disk=${disk}GB" >> $log
    kill $pid
    exit 1
  fi
  sleep 20
done
echo "$(date '+%H:%M:%S') pid $pid finished normally" >> $log
