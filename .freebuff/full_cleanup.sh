#!/bin/bash
# Full cleanup of leftover Mininet state
echo "12345678" | sudo -S mn -c 2>&1
echo "12345678" | sudo -S killall -9 python3 mn iperf 2>/dev/null
echo "12345678" | sudo -S ip link | grep -oP 's\d+-eth\d+|h\d+-eth\d+' | while read intf; do
    echo "12345678" | sudo -S ip link delete "$intf" 2>/dev/null
done
echo "12345678" | sudo -S ovs-vsctl -- --if-exists del-br s1 -- --if-exists del-br s2 -- --if-exists del-br s3 -- --if-exists del-br s4 -- --if-exists del-br s5 -- --if-exists del-br s6 -- --if-exists del-br s7 -- --if-exists del-br s8 -- --if-exists del-br s9 -- --if-exists del-br s10 -- --if-exists del-br s11 -- --if-exists del-br s12 -- --if-exists del-br s13 -- --if-exists del-br s14 2>/dev/null
echo "12345678" | sudo -S ip netns delete $(echo "12345678" | sudo -S ip netns list 2>/dev/null | awk '{print $1}') 2>/dev/null
echo "CLEAN_DONE"
