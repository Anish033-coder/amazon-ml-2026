#!/bin/zsh
# Read-only laptop health log during a long run: one line per interval until <pid> exits.
#   tools/health_log.sh <pid> [interval_seconds=60]
pid=$1
every=${2:-60}
log=/Users/anishkumawat/amazon-ml-2026/experiments/health.log
echo "time      mem_avail%  swap_MB  load1  thermal_warn  perf_warn  battery%  charging  free_disk_GB  job_RSS_GB" >> $log
while kill -0 $pid 2>/dev/null; do
  mem=$(sysctl -n kern.memorystatus_level)
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  therm=$(pmset -g therm 2>/dev/null)
  tw=$(echo "$therm" | grep -qi "No thermal warning" && echo none || echo "$therm" | grep -i "thermal" | head -1 | tr -s ' ' | cut -c1-20)
  pw=$(echo "$therm" | grep -qi "No performance warning" && echo none || echo "$therm" | grep -i "performance" | head -1 | tr -s ' ' | cut -c1-20)
  batt=$(pmset -g batt | grep -o '[0-9]*%' | head -1)
  chg=$(pmset -g batt | grep -q "AC Power" && echo AC || echo battery)
  disk=$(df -g ~ | tail -1 | awk '{print $4}')
  rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
  printf "%s  %9s  %7s  %5s  %12s  %9s  %8s  %8s  %12s  %10s\n" "$(date '+%H:%M:%S')" "$mem" "$swap" "$load" "$tw" "$pw" "$batt" "$chg" "$disk" "$rss" >> $log
  sleep $every
done
echo "$(date '+%H:%M:%S') job $pid finished" >> $log
