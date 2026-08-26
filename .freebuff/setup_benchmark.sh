#!/bin/bash
PW="12345678"

echo "=== [1] Stop Mininet ==="
echo $PW | sudo -S mn -c 2>/dev/null
sleep 2

echo "=== [2] Verify ONOS is running ==="
ONOS_RUNNING=$(echo $PW | sudo -S docker ps 2>/dev/null | grep onos | wc -l)
if [ "$ONOS_RUNNING" = "0" ]; then
  echo "Starting ONOS 1.15.0..."
  echo $PW | sudo -S docker run -d --name onos \
    -p 8181:8181 -p 6653:6653 \
    onosproject/onos:1.15.0 2>&1
  echo "Waiting 40s for ONOS to fully boot..."
  sleep 40
else
  echo "ONOS already running"
  sleep 5
fi

echo "=== [3] Activate apps via Karaf CLI ==="
echo $PW | sudo -S docker exec onos /onos/bin/onos client 2>/dev/null <<'KARAF'
feature:install onos-app-openflow-base
feature:install onos-app-fwd
feature:install onos-app-proxyarp
exit
KARAF
sleep 5

echo "=== [4] Verify ONOS REST ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Devices: {len(d.get(\"devices\",[]))}')
" 2>/dev/null || echo "REST not ready"

echo "=== [5] Start Mininet (background via tmux) ==="
echo $PW | sudo -S tmux kill-session -t mn 2>/dev/null
echo $PW | sudo -S tmux new-session -d -s mn "python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653" 2>&1
echo "Mininet starting in tmux session 'mn'..."
sleep 20

echo "=== [6] Force OF1.3 on all bridges ==="
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set bridge "$br" protocols=OpenFlow13 2>/dev/null
done
echo "OF1.3 forced on $(echo $BRIDGES | wc -w) bridges"

echo "=== [7] Reconnect to ONOS ==="
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set-controller "$br" tcp:127.0.0.1:6653 2>/dev/null
done
sleep 10

echo "=== [8] Check ONOS devices ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
devs = d.get('devices',[])
print(f'Devices: {len(devs)}')
for dv in devs[:3]:
    print(f'  {dv.get(\"id\",\"?\")}  avail={dv.get(\"available\",\"?\")}')
" 2>/dev/null

curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
links = d.get('links',[])
print(f'Links: {len(links)}')
" 2>/dev/null

curl -s -u karaf:karaf http://localhost:8181/onos/v1/hosts 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
hosts = d.get('hosts',[])
print(f'Hosts: {len(hosts)}')
" 2>/dev/null

echo "=== [9] Ping test ==="
echo $PW | sudo -S ping -c 2 -W 2 10.0.0.2 2>&1 | tail -3

echo ""
echo "=== SETUP DONE ==="
