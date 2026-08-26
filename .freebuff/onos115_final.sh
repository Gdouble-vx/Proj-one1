#!/bin/bash
PW="12345678"
KARAF="/root/onos/apache-karaf-3.0.8/bin/client"

echo "=== Stop everything ==="
echo $PW | sudo -S mn -c 2>/dev/null
echo $PW | sudo -S docker rm -f onos 2>/dev/null
sleep 3

echo "=== Start ONOS 1.15.0 ==="
echo $PW | sudo -S docker run -d --name onos \
  -p 8181:8181 -p 6653:6653 \
  onosproject/onos:1.15.0 2>&1
echo "Waiting 60s for ONOS to fully boot..."
sleep 60

echo "=== Check ONOS status ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/cluster/nodes 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
nodes = d.get('nodes',[])
print(f'Cluster nodes: {len(nodes)}')
" 2>/dev/null || echo "Cluster check failed"

echo "=== List available features ==="
echo $PW | sudo -S docker exec onos $KARAF "feature:list -i" 2>/dev/null | grep -i openflow | head -5
echo $PW | sudo -S docker exec onos $KARAF "feature:list -i" 2>/dev/null | grep -i "onos-app" | head -10

echo "=== Install features ==="
echo $PW | sudo -S docker exec onos $KARAF "feature:install onos-app-openflow-base" 2>&1 | tail -5
sleep 5
echo $PW | sudo -S docker exec onos $KARAF "feature:install onos-app-fwd" 2>&1 | tail -5
sleep 5
echo $PW | sudo -S docker exec onos $KARAF "feature:install onos-app-proxyarp" 2>&1 | tail -5
sleep 5

echo "=== Verify apps activated ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/applications 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
apps = d.get('apps',[])
active = [a.get('id','') for a in apps if a.get('active',False)]
print(f'Total apps: {len(apps)}, Active: {len(active)}')
for a in active[:10]:
    print(f'  {a}')
" 2>/dev/null

echo "=== Start Mininet ==="
echo $PW | sudo -S tmux kill-session -t mn 2>/dev/null
echo $PW | sudo -S tmux new-session -d -s mn "python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653" 2>&1
echo "Mininet starting..."
sleep 20

echo "=== Force OF1.3 + reconnect ==="
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set bridge "$br" protocols=OpenFlow13 2>/dev/null
  echo $PW | sudo -S ovs-vsctl set-controller "$br" tcp:127.0.0.1:6653 2>/dev/null
done
echo "OF1.3 + reconnect on $(echo $BRIDGES | wc -w) bridges"
sleep 15

echo "=== Check ONOS ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin); print(f'Devices: {len(d.get(\"devices\",[]))}')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin); print(f'Links: {len(d.get(\"links\",[]))}')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/hosts 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin); print(f'Hosts: {len(d.get(\"hosts\",[]))}')" 2>/dev/null

echo "=== Ping ==="
echo $PW | sudo -S ping -c 2 -W 2 10.0.0.2 2>&1 | tail -3

echo "=== Iperf ==="
echo $PW | sudo -S python3 -c "
import subprocess, time
h1 = subprocess.run('pgrep -f mininet:h1', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
h2 = subprocess.run('pgrep -f mininet:h2', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
subprocess.run(f'sudo nsenter -t {h2} -n iperf -s -D', shell=True, timeout=5)
time.sleep(1)
r = subprocess.run(f'sudo nsenter -t {h1} -n iperf -c 10.0.0.2 -t 3 -f m', shell=True, capture_output=True, text=True, timeout=15)
print(r.stdout[-300:] if r.stdout else 'iperf failed')
" 2>/dev/null

echo "=== DONE ==="
