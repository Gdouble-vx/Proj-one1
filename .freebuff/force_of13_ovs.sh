#!/bin/bash
PW="12345678"

echo "=== Kill old Mininet ==="
echo $PW | sudo -S screen -S mn -X quit 2>/dev/null
echo $PW | sudo -S mn -c 2>/dev/null
sleep 3

echo "=== Start Mininet ==="
echo $PW | sudo -S screen -dmS mn /usr/bin/python3 /home/ino/real_world_topologies_for_mininet.py --topo nsfnet --ip 127.0.0.1 --port 6653
echo "Waiting 20s for Mininet to create bridges..."
sleep 20

echo "=== Force OF1.3 on all bridges ==="
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
echo "Found bridges: $(echo $BRIDGES | wc -w)"
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set bridge $br protocols=OpenFlow13 2>/dev/null
  echo "  Set $br to OpenFlow13"
done

echo "=== Restart OpenFlow connections ==="
# Disconnect and reconnect to force OF1.3 negotiation
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl del-controller $br 2>/dev/null
  echo $PW | sudo -S ovs-vsctl set-controller $br tcp:127.0.0.1:6653 2>/dev/null
done
echo "Controllers reset. Waiting 20s..."
sleep 20

echo "=== Check ONOS ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"devices\",[]))} devices')" 2>/dev/null
curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d.get(\"links\",[]))} links')" 2>/dev/null

echo "=== OF logs ==="
echo $PW | sudo -S docker logs onos 2>&1 | grep -iE "OFChannel|version|1\.0|1\.3|1\.5|negotiat|connect|disconnect" | tail -15

echo "=== Ping test ==="
H1_PID=$(echo $PW | sudo -S ps aux 2>/dev/null | grep "mininet:h1" | grep -v grep | awk '{print $2}' | head -1)
echo "h1 PID=$H1_PID"
echo $PW | sudo -S nsenter -t $H1_PID -n ping -c 3 -W 3 10.0.0.2 2>&1
