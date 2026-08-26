#!/bin/bash
PW="12345678"

echo "=== Stop everything ==="
echo $PW | sudo -S mn -c 2>/dev/null
echo $PW | sudo -S docker rm -f onos 2>/dev/null
sleep 3

echo "=== Start ONOS 2.7.0 ==="
echo $PW | sudo -S docker run -d --name onos \
  -p 8181:8181 -p 6653:6653 \
  onosproject/onos:2.7.0 2>&1
echo "Waiting 40s for ONOS to boot..."
sleep 40

echo "=== Activate apps via REST ==="
for app in org.onosproject.openflow-base org.onosproject.fwd org.onosproject.proxyarp; do
  RESP=$(curl -s -o /dev/null -w "%{http_code}" -X POST -u karaf:karaf \
    -H "Content-Type: application/json" \
    "http://localhost:8181/onos/v1/applications/$app/active" 2>/dev/null)
  echo "  $app → HTTP $RESP"
done
sleep 5

echo "=== Force OF1.3 on ALL bridges ==="
# Stop mininet first
echo $PW | sudo -S mn -c 2>/dev/null
sleep 2

# Start Mininet
echo $PW | sudo -S tmux kill-session -t mn 2>/dev/null
echo $PW | sudo -S tmux new-session -d -s mn "python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653" 2>&1
echo "Mininet starting..."
sleep 20

# Force OF1.3
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set bridge "$br" protocols=OpenFlow13 2>/dev/null
  echo $PW | sudo -S ovs-vsctl set-controller "$br" tcp:127.0.0.1:6653 2>/dev/null
done
echo "OF1.3 forced + reconnected $(echo $BRIDGES | wc -w) bridges"
sleep 15

echo "=== Check ONOS devices ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
devs = d.get('devices',[])
print(f'Devices: {len(devs)}')
" 2>/dev/null

curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
links = d.get('links',[])
print(f'Links: {len(links)}')
" 2>/dev/null

echo "=== Ping test ==="
echo $PW | sudo -S ping -c 2 -W 2 10.0.0.2 2>&1 | tail -3

echo "=== Iperf test ==="
echo $PW | sudo -S python3 -c "
import subprocess, time
h1 = subprocess.run('pgrep -f mininet:h1', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
h2 = subprocess.run('pgrep -f mininet:h2', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
print(f'h1={h1} h2={h2}')
subprocess.run(f'sudo nsenter -t {h2} -n iperf -s -D', shell=True, timeout=5)
time.sleep(1)
r = subprocess.run(f'sudo nsenter -t {h1} -n iperf -c 10.0.0.2 -t 3 -f m', shell=True, capture_output=True, text=True, timeout=15)
print(r.stdout[-300:] if r.stdout else 'failed')
" 2>/dev/null

echo ""
echo "=== DONE ==="
