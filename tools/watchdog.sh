#!/bin/zsh
# Laptop safety guard for long runs: stops the pipeline if memory swap or free disk
# goes past safe limits, instead of letting macOS struggle.
#   tools/watchdog.sh <pid> [max_swap_mb=8000] [min_free_disk_gb=5]
pid=$1
max_swap=${2:-8000}
min_disk=${3:-5}
log=/Users/anishkumawat/amazon-ml-2026/experiments/watchdog.log
while kill -0 $pid 2>/dev/null; do
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
  disk=$(df -g ~ | tail -1 | awk '{print $4}')
  if [ "$swap" -gt "$max_swap" ] || [ "$disk" -lt "$min_disk" ]; then
    echo "$(date '+%H:%M:%S') STOPPED pid $pid: swap=${swap}MB free_disk=${disk}GB" >> $log
    kill $pid
    exit 1
  fi
  sleep 30
done
echo "$(date '+%H:%M:%S') pid $pid finished normally" >> $log
