#!/bin/bash
# Usage: ./runq.sh jobs.txt  — runs each line (train.py args) keeping at most 3 jobs in flight (1 thread each).
P=/home/user/pcgb-venv/bin/python; LOGD=${LOGD:-runs/_stdout}; mkdir -p $LOGD
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  PYTHONUNBUFFERED=1 $P train.py $line > "$LOGD/$(echo $line | tr ' -' '_' | tr -s '_').txt" 2>&1 &
  sleep 2
done < "$1"
wait
echo "QUEUE_DONE $1"
