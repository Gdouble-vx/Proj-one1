#!/usr/bin/env python3
"""bench_real.py — Real ONOS+Mininet benchmark (SVs1)."""
import subprocess, json, time, os

PW = "12345678"
ONOS_USER = "onos:rocks"
ONOS_API = "http://localhost:8181/onos/v1"
HOME = "/home/ino"

def sudo(cmd, timeout=30):
    r = subprocess.run(f"echo {PW} | sudo -S bash -c '{cmd}'",
                       shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr

def curl_api(method, path):
    r = subprocess.run(f"curl -s -u {ONOS_USER} -X {method} '{ONOS_API}{path}'",
                       shell=True, capture_output=True, text=True, timeout=10)
    return r.stdout

def main():
    print("=" * 60)
    print("  Real Benchmark: OSPF vs PPO+GNN")
    print("  ONOS 2.7.0 + Mininet + NSFNET (14N/21L)")
    print("=" * 60)

    # 1. Clean
    print("\n[1/6] Cleaning...")
    sudo("mn -c 2>/dev/null", 10)
    sudo("docker rm -f onos 2>/dev/null", 10)
    time.sleep(2)

    # 2. Start ONOS
    print("[2/7] Starting ONOS 2.7.0...")
    out = sudo("docker run -d --name onos -p 8181:8181 -p 6653:6653 onosproject/onos:2.7.0 2>&1")
    print(f"  {out.strip()[:80]}")

    print("  Waiting for ONOS...")
    for i in range(24):
        time.sleep(5)
        resp = curl_api("GET", "/devices")
        # ONOS is ready when it returns a JSON (not 500 error)
        if resp.strip().startswith('{') and 'code' not in resp[:50]:
            print(f"  ONOS ready! ({(i+1)*5}s)")
            break
        print(f"  ({(i+1)*5}s)")

    # 3. Activate apps
    print("[3/7] Activating ONOS apps...")
    for app in ["org.onosproject.drivers.openflow", "org.onosproject.openflow-base",
                 "org.onosproject.openflow", "org.onosproject.proxyarp"]:
        curl_api("POST", f"/apps/{app}/active")
        print(f"  ✓ {app.split('.')[-1]}")
    time.sleep(5)

    print("  Waiting extra 15s for ONOS apps to fully initialize...")
    time.sleep(15)

    # 4. Start Mininet
    print("[4/7] Starting Mininet NSFNET...")
    mn_code = '''
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.log import setLogLevel
import time, signal
setLogLevel('warning')

class NSFNETTopo(Topo):
    def __init__(self, **kw):
        super().__init__(**kw)
        s = {}
        for i in range(1,15):
            s[i] = self.addSwitch(f's{i}', dpid=f'{i:016x}', protocols='OpenFlow13')
        for i in range(1,15):
            self.addHost(f'h{i}', ip=f'10.0.0.{i}/24', mac=f'00:00:00:00:00:{i:02x}')
            self.addLink(f'h{i}', s[i], bw=100)
        for u,v,bw in [(1,2,150),(1,3,80),(1,8,100),(2,3,20),(2,7,15),(3,4,15),
                        (4,5,100),(4,6,200),(5,6,150),(5,7,150),(6,13,15),(6,14,80),
                        (7,8,100),(8,9,20),(9,10,150),(9,12,200),(10,11,100),
                        (10,13,100),(11,12,200),(11,14,15),(12,13,80)]:
            self.addLink(s[u], s[v], bw=bw, delay='2ms')

net = Mininet(topo=NSFNETTopo(), switch=OVSKernelSwitch, link=TCLink,
              controller=RemoteController('c0', ip='127.0.0.1', port=6653),
              autoSetMacs=True, autoStaticArp=True)
net.start()
print(f'MN_OK {len(net.switches)}sw {len(net.hosts)}h')
time.sleep(15)
r = net.pingAll()
print(f'PING {r}')
signal.pause()
'''
    mn_path = f"{HOME}/mn_nsfnet.py"
    with open(mn_path, 'w') as f:
        f.write(mn_code)
    sudo(f"nohup python3 {mn_path} > /tmp/mn_out.log 2>&1 &")
    print("  Waiting 45s for Mininet + switch connection...")
    time.sleep(45)

    # 5. Verify switches
    print("[5/7] Checking switches...")
    resp = curl_api("GET", "/devices")
    try:
        devs = json.loads(resp).get('devices', [])
        avail = [d for d in devs if d.get('available')]
        print(f"  Connected: {len(avail)} switches")
    except:
        print(f"  Parse error")

    # 6. OSPF iperf
    print("[6/7] Running OSPF iperf...")
    ospf_flows = []
    pairs = [(f'h{i}', f'h{j}') for i in range(1, 8) for j in range(i+1, min(i+3, 8))]

    for src, dst in pairs:
        dst_num = ord(dst[1]) - 96
        r = subprocess.run(f"pgrep -f 'mininet:{src}'", shell=True, capture_output=True, text=True)
        src_pid = r.stdout.strip().split('\n')[0] if r.stdout.strip() else None
        r = subprocess.run(f"pgrep -f 'mininet:{dst}'", shell=True, capture_output=True, text=True)
        dst_pid = r.stdout.strip().split('\n')[0] if r.stdout.strip() else None
        if not src_pid or not dst_pid:
            continue
        # Start server in background (nohup + &)
        sudo(f"nohup nsenter -t {dst_pid} -n iperf -s -1 -f m > /dev/null 2>&1 &", 5)
        time.sleep(1)
        out = sudo(f"nsenter -t {src_pid} -n iperf -c 10.0.0.{dst_num} -t 3 -P 2 -f m 2>&1", 20)
        bw = 0.0
        for line in out.split('\n'):
            if 'bits/sec' in line and 'SUM' not in line.upper():
                try:
                    parts = line.split()
                    bw = float(parts[-2])
                    if 'Gbits' in parts[-1]:
                        bw *= 1000
                except:
                    pass
        if bw > 0:
            ospf_flows.append({'src': src, 'dst': dst, 'mbps': round(bw, 2)})
            print(f"  {src}->{dst}: {bw:.1f} Mbps")

    avg_ospf = sum(f['mbps'] for f in ospf_flows) / len(ospf_flows) if ospf_flows else 0
    print(f"  OSPF Average: {avg_ospf:.1f} Mbps ({len(ospf_flows)} flows)")

    # 7. Load PPO+GNN weights
    print("[7/7] Loading PPO+GNN weights...")
    try:
        with open(f'{HOME}/ppognn_weights.json') as f:
            ppognn_rules = json.load(f)
        print(f"  Got {len(ppognn_rules)} rules")
        for rule in ppognn_rules:
            u = int(rule['src'].split(':')[-1], 16)
            v = int(rule['dst'].split(':')[-1], 16)
            w = rule['weight']
            m = "❌ avoid" if w > 30 else "✅ prefer" if w < 5 else "~ neutral"
            print(f"    s{u}-s{v}: {w:.1f} {m}")
    except Exception as e:
        print(f"  Error: {e}")
        ppognn_rules = None

    # Save
    results = {
        'topology': 'NSFNET (14 nodes, 21 links)',
        'controller': 'ONOS 2.7.0',
        'ospf_baseline': {'flows': ospf_flows, 'avg_mbps': round(avg_ospf, 2)},
        'ppognn_weights': ppognn_rules,
    }
    with open(f'{HOME}/real_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  OSPF Average: {avg_ospf:.1f} Mbps")
    print(f"  Saved: {HOME}/real_benchmark_results.json")
    print("=" * 60)

if __name__ == '__main__':
    main()
