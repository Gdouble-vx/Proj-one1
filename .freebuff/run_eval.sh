#!/bin/bash
cd /home/ino/sdn-ai-brain
source venv/bin/activate
pkill -f fine_tune_sdn_agent.py 2>/dev/null
sleep 2
setsid nohup python -u fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 21 \
    --base-model ppo_gnn_sdn_model --eval-only --eval-episodes 5 \
    > /tmp/eval_100k.log 2>&1 < /dev/null &
echo "EVAL_PID: $!"
sleep 3
ps ax | grep fine_tune | grep -v grep
echo "--- log head ---"
head -8 /tmp/eval_100k.log
