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

echo "Waiting 60s for ONOS to fully boot..."
sleep 60

echo "=== Try Karaf SSH activation ==="
# Use docker exec to run karaf client
echo $PW | sudo -S docker exec onos /root/onos/apache-karaf-4.2.9/bin/client 2>&1 <<'EOF'
feature:install onos-app-openflow-base
EOF
sleep 5

echo $PW | sudo -S docker exec onos /root/onos/apache-karaf-4.2.9/bin/client 2>&1 <<'EOF'
feature:install onos-app-fwd
EOF
sleep 5

echo $PW | sudo -S docker exec onos /root/onos/apache-karaf-4.2.9/bin/client 2>&1 <<'EOF'
feature:install onos-app-proxyarp
EOF
sleep 5

echo "=== Check apps ==="
echo $PW | sudo -S docker exec onos /root/onos/apache-karaf-4.2.9/bin/client 2>&1 <<'EOF'
app:list -a
EOF

echo "=== Try REST with empty body ==="
curl -s -o /dev/null -w "HTTP %{http_code}" -X POST -u karaf:karaf \
  "http://localhost:8181/onos/v1/applications/org.onosproject.openflow-base/active" 2>/dev/null
echo ""

echo "=== Check ONOS devices ==="
curl -s -u karaf:karaf http://localhost:8181/onos/v1/devices 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin); print(f'Devices: {len(d.get(\"devices\",[]))}')" 2>/dev/null
