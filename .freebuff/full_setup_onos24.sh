#!/bin/bash
PW="12345678"

echo "=== [1/6] Kill old containers/Mininet ==="
echo $PW | sudo -S docker rm -f onos 2>/dev/null
echo $PW | sudo -S mn -c 2>/dev/null
echo $PW | sudo -S screen -S mn -X quit 2>/dev/null
sleep 3

echo "=== [2/6] Start ONOS 2.4.0 ==="
echo $PW | sudo -S docker run -d --name onos \
  -p 8181:8181 -p 6653:6653 -p 8101:8101 \
  -e JAVA_MIN_MEM=512m -e JAVA_MAX_MEM=2048m \
  onosproject/onos:2.4.0

echo "Waiting 120s for ONOS bootstrap..."
for i in $(seq 1 12); do
  sleep 10
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -u karaf:karaf http://localhost:8181/onos/v1/cluster/nodes 2>/dev/null)
  echo "  [$((i*10))s] REST=$CODE"
  if [ "$CODE" = "200" ]; then
    echo "ONOS READY!"
    break
  fi
done

echo "=== [3/6] Activate apps ==="
for app in org.onosproject.openflow org.onosproject.fwd org.onosproject.proxyarp; do
  RESP=$(curl -s -w "%{http_code}" -o /dev/null -u karaf:karaf -X POST "http://localhost:8181/onos/v1/applications/$app/active" 2>/dev/null)
  echo "  $app: HTTP $RESP"
done
sleep 15
echo "=== OF port ==="
echo $PW | sudo -S ss -tlnp 2>/dev/null | grep 6653

echo "=== [4/6] Start Mininet ==="
echo $PW | sudo -S screen -dmS mn /usr/bin/python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653
echo "Waiting 40s for Mininet..."
sleep 40

echo "=== Mininet output ==="
echo $PW | sudo -S screen -S mn -X hardcopy /tmp/mn_out.txt 2>/dev/null
echo $PW | sudo -S cat /tmp/mn_out.txt 2>&1 | grep -E "Testing|Ping|h1|Starting|Adding" | head -10

echo ""
echo "=== [5/6] Check ONOS ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); devs=d.get('devices',[]); print(f'Devices: {len(devs)}')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Links: {len(d.get(\"links\",[]))}')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/hosts 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Hosts: {len(d.get(\"hosts\",[]))}')" 2>/dev/null

echo ""
echo "=== [6/6] Ping test ==="
H1_PID=$(echo $PW | sudo -S ps aux 2>/dev/null | grep "mininet:h1" | grep -v grep | awk '{print $2}' | head -1)
echo "h1 PID=$H1_PID"
if [ -n "$H1_PID" ]; then
  echo $PW | sudo -S nsenter -t $H1_PID -n ping -c 3 -W 3 10.0.0.2 2>&1
fi
