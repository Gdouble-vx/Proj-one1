#!/bin/bash
PW="12345678"
echo "=== Start Mininet ==="
echo $PW | sudo -S screen -dmS mn /usr/bin/python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653
echo "Waiting 45s..."
sleep 45

echo "=== Mininet output ==="
echo $PW | sudo -S screen -S mn -X hardcopy /tmp/mn_out.txt 2>/dev/null
echo $PW | sudo -S cat /tmp/mn_out.txt 2>&1 | grep -E "Testing|Ping|h1|Starting|Adding|MN" | head -15

echo "=== ONOS Devices ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"devices\",[]))} devices')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"links\",[]))} links')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/hosts 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"hosts\",[]))} hosts')" 2>/dev/null

echo "=== OF log after Mininet ==="
echo $PW | sudo -S docker logs onos 2>&1 | tail -20 | grep -iE "switch|connect|device|master|hello" | head -10

echo "=== Ping test ==="
H1_PID=$(echo $PW | sudo -S ps aux 2>/dev/null | grep "mininet:h1" | grep -v grep | awk '{print $2}' | head -1)
echo "h1 PID=$H1_PID"
echo $PW | sudo -S nsenter -t $H1_PID -n ping -c 3 -W 3 10.0.0.2 2>&1
