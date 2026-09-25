#!/bin/zsh
# Run one pipeline command as its own process with the laptop safety guards attached
# (watchdog, health log, stay-awake), and wait for it.
#   tools/run_guarded.sh <log_file> <src.run arguments...>
root=/Users/anishkumawat/amazon-ml-2026
log=$1; shift
cd $root/code/business_entity_resolution
PYTHONPATH=$root/code/business_entity_resolution $root/.venv/bin/python -m src.run "$@" >> $log 2>&1 &
pid=$!
$root/tools/watchdog.sh $pid &
$root/tools/health_log.sh $pid 60 &
caffeinate -i -w $pid &
wait $pid
