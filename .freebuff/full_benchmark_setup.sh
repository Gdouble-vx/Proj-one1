#!/bin/bash
PW="12345678"

echo "=== [1] Stop old Mininet ==="
echo $PW | sudo -S mn -c 2>/dev/null
sleep 2

echo "=== [2] Stop old ONOS Docker ==="
echo $PW | sudo -S docker rm -f onos 2>/dev/null
sleep 3

echo "=== [3] Start ONOS 1.15 Docker ==="
echo $PW | sudo -S docker run -d --name onos \
  -p 8181:8181 -p 6653:6653 \
  -e ONOS_CLIENT_IP=127.0.0.1 \
  onosproject/onos:1.15.11 2>&1
echo "Waiting 30s for ONOS to boot..."
sleep 30

echo "=== [4] Verify ONOS REST ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | head -1
echo ""

echo "=== [5] Activate OpenFlow apps ==="
for app in org.onosproject.openflow-base org.onosproject.fwd org.onosproject.proxyarp; do
  curl -s -X POST -u karaf:karaf -H "Content-Type: application/json" \
    "http://localhost:8181/onos/v1/applications/$app/active" 2>/dev/null
  echo "Activated: $app"
done
sleep 5

echo "=== [6] Start Mininet with asymmetric NSFNET ==="
echo $PW | sudo -S python3 /home/ino/real_world_topologies_for_mininet.py \
  --topo nsfnet --ip 127.0.0.1 --port 6653 > /tmp/mn_startup.log 2>&1 &
MN_PID=$!
echo "Mininet PID: $MN_PID"
sleep 15

echo "=== [7] Force OF1.3 on all bridges ==="
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set bridge "$br" protocols=OpenFlow13 2>/dev/null
done
echo "OF1.3 forced on all bridges"

echo "=== [8] Reconnect to ONOS ==="
sleep 5
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set-controller "$br" tcp:127.0.0.1:6653 2>/dev/null
done
echo "Reconnected all bridges to ONOS"
sleep 10

echo "=== [9] Verify ONOS devices ==="
DEVICES=$(curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null)
echo "$DEVICES" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    devs = d.get('devices', [])
    print(f'Devices: {len(devs)}')
except:
    print('Parse failed')
" 2>/dev/null

LINKS=$(curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null)
echo "$LINKS" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    links = d.get('links', [])
    print(f'Links: {len(links)}')
except:
    print('Parse failed')
" 2>/dev/null

echo "=== [10] Test ping ==="
echo $PW | sudo -S python3 -c "
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
# Just test via ip commands
import subprocess
result = subprocess.run(['ping', '-c', '2', '-W', '2', '10.0.0.2'], capture_output=True, text=True, timeout=10)
print(result.stdout[-200:] if result.stdout else 'ping failed')
" 2>/dev/null

echo "=== [11] Test iperf ==="
echo $PW | sudo -S python3 -c "
import subprocess
# Find h1 and h2 PIDs
h1_pid = subprocess.run('pgrep -f mininet:h1', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
h2_pid = subprocess.run('pgrep -f mininet:h2', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
print(f'h1 PID: {h1_pid}, h2 PID: {h2_pid}')

# Start iperf server on h2
subprocess.run(f'sudo nsenter -t {h2_pid} -n iperf -s -D', shell=True, timeout=5)
import time; time.sleep(1)

# Run iperf client from h1
result = subprocess.run(f'sudo nsenter -t {h1_pid} -n iperf -c 10.0.0.2 -t 3 -f m', 
                       shell=True, capture_output=True, text=True, timeout=15)
print(result.stdout[-300:] if result.stdout else 'iperf failed')
print(result.stderr[-200:] if result.stderr else '')
" 2>/dev/null

echo ""
echo "=== SETUP COMPLETE ==="
echo "ONOS: http://localhost:8181 (karaf/karaf)"
echo "Mininet: running with asymmetric NSFNET topology"
