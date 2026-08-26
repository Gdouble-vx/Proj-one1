#!/bin/bash
PW="12345678"
echo $PW | sudo -S pkill -f pox.py 2>/dev/null
echo $PW | sudo -S mn -c 2>/dev/null
sleep 2
echo $PW | sudo -S nohup python3 /home/ino/pox/pox.py forwarding.l2_pairs --address=0.0.0.0 --port=6653 > /tmp/pox.log 2>&1 &
sleep 5
echo $PW | sudo -S python3 /home/ino/benchmark_standalone.py 2>&1
echo $PW | sudo -S pkill -f pox.py 2>/dev/null
