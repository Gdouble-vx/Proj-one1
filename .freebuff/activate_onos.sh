#!/bin/bash
PW="12345678"
KARAF="/root/onos/apache-karaf-3.0.8/bin/client"

echo "=== Activate OpenFlow apps ==="
echo $PW | sudo -S docker exec onos $KARAF "feature:install onos-app-openflow-base" 2>&1 | tail -3
sleep 3
echo $PW | sudo -S docker exec onos $KARAF "feature:install onos-app-fwd" 2>&1 | tail -3
sleep 3
echo $PW | sudo -S docker exec onos $KARAF "feature:install onos-app-proxyarp" 2>&1 | tail -3
sleep 3

echo "=== Verify ONOS devices ==="
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

echo "=== Force OF1.3 + reconnect ==="
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set-controller "$br" tcp:127.0.0.1:6653 2>/dev/null
done
sleep 10

echo "=== Final device count ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
devs = d.get('devices',[])
print(f'Devices: {len(devs)}')
for dv in devs:
    print(f'  {dv.get(\"id\",\"?\")}')
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

# Find host PIDs
h1_pid = subprocess.run('pgrep -f mininet:h1', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
h2_pid = subprocess.run('pgrep -f mininet:h2', shell=True, capture_output=True, text=True).stdout.strip().split('\n')[0]
print(f'h1={h1_pid} h2={h2_pid}')

# Start iperf server on h2
subprocess.run(f'sudo nsenter -t {h2_pid} -n iperf -s -D', shell=True, timeout=5)
time.sleep(1)

# Run iperf from h1
r = subprocess.run(f'sudo nsenter -t {h1_pid} -n iperf -c 10.0.0.2 -t 3 -f m', shell=True, capture_output=True, text=True, timeout=15)
print(r.stdout[-300:] if r.stdout else 'failed')
" 2>/dev/null

echo ""
echo "=== BENCHMARK READY ==="
