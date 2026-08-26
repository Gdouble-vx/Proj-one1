#!/bin/bash
PW="12345678"
echo "=== ONOS Status ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); devs=d.get('devices',[]); print(f'{len(devs)} devices')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"links\",[]))} links')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/hosts 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"hosts\",[]))} hosts')" 2>/dev/null

echo "=== Bridges ==="
echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null | wc -l

echo "=== Test iperf h1 -> h2 ==="
H1_PID=$(echo $PW | sudo -S ps aux 2>/dev/null | grep "mininet:h1" | grep -v grep | awk '{print $2}' | head -1)
H2_PID=$(echo $PW | sudo -S ps aux 2>/dev/null | grep "mininet:h2" | grep -v grep | awk '{print $2}' | head -1)
echo "h1=$H1_PID h2=$H2_PID"
echo $PW | sudo -S nsenter -t $H2_PID -n iperf -s -D 2>/dev/null
sleep 1
echo $PW | sudo -S nsenter -t $H1_PID -n iperf -c 10.0.0.2 -t 3 -P 1 -f m 2>/dev/null | tail -5
