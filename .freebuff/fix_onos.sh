#!/bin/bash
PW="12345678"
echo $PW | sudo -S docker rm -f onos 2>/dev/null

echo "Trying ONOS versions..."
for TAG in "1.15.0" "1.14.0" "1.13.0" "2.7.0" "latest"; do
  echo "=== Trying onosproject/onos:$TAG ==="
  echo $PW | sudo -S docker pull "onosproject/onos:$TAG" 2>&1 | tail -3
  if [ $? -eq 0 ]; then
    echo "Pulled $TAG successfully"
    echo $PW | sudo -S docker run -d --name onos \
      -p 8181:8181 -p 6653:6653 \
      "onosproject/onos:$TAG" 2>&1
    echo "Tag used: $TAG" > /tmp/onos_tag.txt
    break
  fi
done

echo "Waiting 30s for ONOS to boot..."
sleep 30

echo "=== Verify ONOS ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | head -1
echo ""

echo "=== Activate OpenFlow apps ==="
for app in org.onosproject.openflow-base org.onosproject.fwd org.onosproject.proxyarp; do
  curl -s -X POST -u karaf:karaf -H "Content-Type: application/json" \
    "http://localhost:8181/onos/v1/applications/$app/active" 2>/dev/null
  echo "Activated: $app"
done

echo "=== Force OF1.3 + reconnect ==="
BRIDGES=$(echo $PW | sudo -S ovs-vsctl list-br 2>/dev/null)
for br in $BRIDGES; do
  echo $PW | sudo -S ovs-vsctl set bridge "$br" protocols=OpenFlow13 2>/dev/null
  echo $PW | sudo -S ovs-vsctl set-controller "$br" tcp:127.0.0.1:6653 2>/dev/null
done
echo "Done: $BRIDGES"

sleep 10

echo "=== Final check ==="
DEVICES=$(curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null)
echo "$DEVICES" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Devices: {len(d.get(\"devices\",[]))}')" 2>/dev/null

LINKS=$(curl -s -u karaf:karaf http://localhost:8181/onos/v1/links 2>/dev/null)
echo "$LINKS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Links: {len(d.get(\"links\",[]))}')" 2>/dev/null

echo "=== Ping test ==="
echo $PW | sudo -S ping -c 2 -W 2 10.0.0.2 2>&1 | tail -3
