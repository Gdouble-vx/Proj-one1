#!/bin/bash
export PW="12345678"

echo "=== NSFNET Multi-Flow Benchmark ==="

# Start iperf servers on h2-h5
for i in 2 3 4 5; do
  H_PID=$(ps aux | grep "mininet:h${i}$" | grep -v grep | awk '{print $2}' | head -1)
  echo $PW | sudo -S nsenter -t $H_PID -n iperf -s -D 2>/dev/null
done
sleep 1

# Get host PIDS
H1=$(ps aux | grep "mininet:h1$" | grep -v grep | awk '{print $2}' | head -1)
H6=$(ps aux | grep "mininet:h6$" | grep -v grep | awk '{print $2}' | head -1)
H7=$(ps aux | grep "mininet:h7$" | grep -v grep | awk '{print $2}' | head -1)
H8=$(ps aux | grep "mininet:h8$" | grep -v grep | awk '{print $2}' | head -1)

echo "Hosts: h1=$H1 h6=$H6 h7=$H7 h8=$H8"

# Run 4 flows simultaneously
echo $PW | sudo -S nsenter -t $H1 -n iperf -c 10.0.0.2 -t 5 -P 2 -f m > /tmp/iperf1.txt 2>&1 &
echo $PW | sudo -S nsenter -t $H6 -n iperf -c 10.0.0.3 -t 5 -P 1 -f m > /tmp/iperf2.txt 2>&1 &
echo $PW | sudo -S nsenter -t $H7 -n iperf -c 10.0.0.4 -t 5 -P 1 -f m > /tmp/iperf3.txt 2>&1 &
echo $PW | sudo -S nsenter -t $H8 -n iperf -c 10.0.0.5 -t 5 -P 1 -f m > /tmp/iperf4.txt 2>&1 &

sleep 10
wait

echo "=== Results ==="
TOTAL=0
for i in 1 2 3 4; do
  BW=$(grep "Mbits/sec" /tmp/iperf${i}.txt | tail -1 | awk '{print $(NF-1)}')
  echo "Flow $i: ${BW:-0} Mbps"
done
