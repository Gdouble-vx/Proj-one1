#!/bin/bash
PW="12345678"

echo "=== Stop ONOS Docker (occupying port 6653) ==="
echo $PW | sudo -S docker rm -f onos 2>/dev/null
echo $PW | sudo -S docker stop onos 2>/dev/null
sleep 2

echo "=== Verify port 6653 free ==="
echo $PW | sudo -S ss -tlnp | grep 6653 2>/dev/null || echo "Port 6653 is FREE"

echo "=== Stop Mininet ==="
echo $PW | sudo -S mn -c 2>/dev/null
sleep 2

echo "=== Start POX on port 6653 ==="
echo $PW | sudo -S pkill -f pox.py 2>/dev/null
sleep 1
echo $PW | sudo -S tmux kill-session -t pox 2>/dev/null
echo $PW | sudo -S tmux new-session -d -s pox "cd /home/ino/pox && python3 pox.py forwarding.l2_pairs --address=0.0.0.0 --port=6653 log.level --DEBUG" 2>&1
echo "POX starting..."
sleep 5

echo "=== Verify POX listening ==="
echo $PW | sudo -S ss -tlnp | grep 6653 2>/dev/null

echo "=== Start Mininet ==="
echo $PW | sudo -S tmux kill-session -t mn 2>/dev/null
echo $PW | sudo -S tmux new-session -d -s mn "python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653" 2>&1
echo "Mininet starting..."
sleep 25

echo "=== Check bridges ==="
echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null | wc -l

echo "=== Check OVS controller ==="
echo $PW | sudo -S ovs-vsctl get-controller s1 2>/dev/null

echo "=== POX tmux output ==="
echo $PW | sudo -S tmux capture-pane -t pox -p 2>/dev/null | tail -20

echo "=== Ping test ==="
echo $PW | sudo -S ping -c 3 -W 3 10.0.0.2 2>&1 | tail -5

echo "=== Iperf test ==="
echo $PW | sudo -S python3 -c "
import subprocess, time
h1 = subprocess.run('pgrep -f mininet:h1', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
h2 = subprocess.run('pgrep -f mininet:h2', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
print(f'h1={h1} h2={h2}')
subprocess.run(f'sudo nsenter -t {h2} -n pkill iperf', shell=True, capture_output=True, timeout=3)
time.sleep(0.5)
subprocess.run(f'sudo nsenter -t {h2} -n iperf -s -D', shell=True, capture_output=True, timeout=5)
time.sleep(1)
r = subprocess.run(f'sudo nsenter -t {h1} -n iperf -c 10.0.0.2 -t 3 -f m', shell=True, capture_output=True, text=True, timeout=15)
print(r.stdout[-300:] if r.stdout else 'iperf failed')
" 2>/dev/null

echo "=== DONE ==="
