#!/bin/bash
PW="12345678"
SVS1="192.168.10.165"

run_multi_iperf() {
  # Start 4 concurrent iperf flows through different paths
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h2" | grep -v grep | awk '{print $2}' | head -1) -n iperf -s -D 2>/dev/null
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h3" | grep -v grep | awk '{print $2}' | head -1) -n iperf -s -D 2>/dev/null
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h4" | grep -v grep | awk '{print $2}' | head -1) -n iperf -s -D 2>/dev/null
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h5" | grep -v grep | awk '{print $2}' | head -1) -n iperf -s -D 2>/dev/null
  sleep 1

  # Run 4 flows simultaneously
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h1" | grep -v grep | awk '{print $2}' | head -1) -n iperf -c 10.0.0.2 -t 5 -P 2 -f m &>/tmp/iperf_h1.txt &
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h6" | grep -v grep | awk '{print $2}' | head -1) -n iperf -c 10.0.0.3 -t 5 -P 1 -f m &>/tmp/iperf_h6.txt &
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h7" | grep -v grep | awk '{print $2}' | head -1) -n iperf -c 10.0.0.4 -t 5 -P 1 -f m &>/tmp/iperf_h7.txt &
  $PW | sudo -S nsenter -t $($PW | sudo -S ps aux | grep "mininet:h8" | grep -v grep | awk '{print $2}' | head -1) -n iperf -c 10.0.0.5 -t 5 -P 1 -f m &>/tmp/iperf_h8.txt &

  sleep 8
  wait

  # Sum up throughput
  TOTAL=0
  for f in /tmp/iperf_*.txt; do
    BW=$(grep "Mbits/sec" $f | tail -1 | awk '{print $(NF-1)}')
    TOTAL=$(echo "$TOTAL + $BW" | bc 2>/dev/null || echo "$TOTAL")
    echo "  $(basename $f): ${BW} Mbps"
  done
  echo "  TOTAL: ${TOTAL} Mbps"
}

echo "=== NSFNET Asymmetric Topology - Multi-Flow Benchmark ==="
echo ""

echo "--- OSPF (ONOS fwd, hop-count routing) ---"
sshpass -p 12345678 ssh -o StrictHostKeyChecking=no ino@$SVS1 "
echo 12345678 | sudo -S ovs-vsctl list-br | wc -l
" 2>/dev/null

# Run OSPF benchmark (forwarding enabled by default)
run_multi_iperf

echo ""
echo "=== DONE ==="
