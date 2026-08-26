#!/bin/bash
PW="12345678"
echo $PW | sudo -S pkill -f pox.py 2>/dev/null
echo $PW | sudo -S pkill -f benchmark 2>/dev/null
echo $PW | sudo -S mn -c 2>/dev/null
sleep 2

# Start Mininet in background (with no controller)
echo $PW | sudo -S python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653 > /tmp/mn.log 2>&1 &
sleep 20

# Set standalone mode
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set-fail-mode "$br" standalone 2>/dev/null
done
echo "Set standalone mode on all bridges"
sleep 5

# Run benchmark
echo $PW | sudo -S python3 /home/ino/fast_benchmark.py 2>&1
