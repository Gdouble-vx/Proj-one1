#!/usr/bin/env python3
"""
Install OF rules on ONOS based on PPO+GNN optimized weights
and run iperf comparison with OSPF
"""
import json
import subprocess
import time
import urllib.request
import urllib.error
import base64

ONOS_URL = "http://192.168.10.165:8181"
AUTH = base64.b64encode(b"karaf:karaf").decode()

def onos_request(method, path, data=None):
    url = f"{ONOS_URL}/onos/v1/{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Basic {AUTH}")
    if data:
        req.add_header("Content-Type", "application/json")
        req = urllib.request.Request(url, data=json.dumps(data).encode(), method=method)
        req.add_header("Authorization", f"Basic {AUTH}")
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read()) if resp.read() else {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

def onos_get(path):
    url = f"{ONOS_URL}/onos/v1/{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Basic {AUTH}")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def install_flow(switch_dpid, priority, match_fields, actions):
    url = f"{ONOS_URL}/onos/v1/flows/{switch_dpid}"
    flow = {
        "priority": priority,
        "timeout": 0,
        "isPermanent": True,
        "deviceId": switch_dpid,
        "tableId": 0,
        "treatment": {"instructions": actions},
        "selector": {"criteria": match_fields}
    }
    data = json.dumps({"flow": flow}).encode()
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header("Authorization", f"Basic {AUTH}")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:
        return str(e)

def get_port_mapping():
    """Get port mappings from ONOS links."""
    data = onos_get("links")
    links = data.get("links", [])
    # Build port map: (src_device, dst_device) -> port_on_src
    # ONOS format: {"src": {"port": "4", "device": "of:0000000000000003"},
    #              "dst": {"port": "2", "device": "of:0000000000000004"}}
    port_map = {}
    for link in links:
        src = link.get("src", {})
        dst = link.get("dst", {})
        src_device = src.get("device", "")
        src_port = int(src.get("port", 0))
        dst_device = dst.get("device", "")
        dst_port = int(dst.get("port", 0))
        if src_device and dst_device:
            port_map[(src_device, dst_device)] = src_port
            port_map[(dst_device, src_device)] = dst_port
    return port_map

def get_host_ports():
    """Get host port mappings from ONOS hosts."""
    data = onos_get("hosts")
    hosts = data.get("hosts", [])
    # ONOS format: {"id": "00:00:00:00:00:02/None", "mac": "00:00:00:00:00:02",
    #               "ipAddresses": [], "locations": [{"elementId": "of:...", "port": "1"}]}
    host_ports = {}
    for h in hosts:
        mac = h.get("mac", "")
        ips = h.get("ipAddresses", [])
        locations = h.get("locations", [])
        if not ips:
            # Try to derive IP from MAC (00:00:00:00:00:02 -> 10.0.0.2)
            try:
                mac_num = int(mac.replace(":", ""), 16)
                if mac_num <= 14:
                    ips = [f"10.0.0.{mac_num}"]
            except:
                pass
        if locations:
            loc = locations[0]
            dpid = loc.get("elementId", "")
            port = int(loc.get("port", 0))
            for ip in ips:
                host_ports[ip] = {"dpid": dpid, "port": port}
    return host_ports

def setup_ppo_rules(port_map, host_ports):
    """Install PPO+GNN optimized flow rules."""
    print("\nInstalling PPO+GNN flow rules...")
    
    # Load PPO weights and compute paths
    import sys
    sys.path.insert(0, '/home/ino')
    from network_sim import NetworkSimulator
    import numpy as np
    
    sim = NetworkSimulator(topology='nsfnet', seed=42)
    sim.reset = lambda **kw: None
    sim.step_count = 0
    
    # Load trained weights
    with open('/home/ino/finetune_results.json', 'r') as f:
        results = json.load(f)
    
    ppo_weights = np.array(results['ppo_weights'], dtype=np.float32)
    paths = []
    for src, dst, demand in sim.flows:
        path = sim.dijkstra_path(src, dst, ppo_weights)
        paths.append(path)
    
    # Install rules for each flow
    rules_installed = 0
    for flow_idx, (src, dst, demand) in enumerate(sim.flows):
        path = paths[flow_idx]
        if not path:
            continue
        
        src_ip = f"10.0.0.{src+1}"
        dst_ip = f"10.0.0.{dst+1}"
        
        for i, link_idx in enumerate(path):
            u = int(sim.edges_u[link_idx])
            v = int(sim.edges_v[link_idx])
            
            # Get DPIDs (ONOS format: of:0000000000000001)
            src_dpid = f"of:{u+1:016x}"
            dst_dpid = f"of:{v+1:016x}"
            
            # Get port from port map
            port = port_map.get((src_dpid, dst_dpid))
            
            if port is None:
                print(f"  WARNING: No port mapping for {src_dpid} -> {dst_dpid}")
                continue
            
            # Get host port for match
            host_port = host_ports.get(src_ip, {}).get("port", 1) if i == 0 else None
            
            # Install flow rule: match on input port, output to next hop
            match = [{"type": "IN_PORT", "port": host_port}] if host_port else [{"type": "IN_PORT", "port": port - 1}]
            result = install_flow(
                src_dpid, 100,
                match,
                [{"type": "OUTPUT", "port": port}]
            )
            print(f"  {src_ip}->{dst_ip} sw{s+1}: in={match[0]['port']} out={port} -> {result}")
            rules_installed += 1
    
    print(f"\nInstalled {rules_installed} flow rules")
    return rules_installed

def run_iperf_test(src_host, dst_host, duration=5):
    """Run iperf from src_host to dst_host on SVs1."""
    cmd = [
        "sshpass", "-p", "12345678",
        "ssh", "-o", "StrictHostKeyChecking=no",
        "ino@192.168.10.165",
        f"echo 12345678 | sudo -S nsenter -t $(echo 12345678 | sudo -S ps aux | grep mininet:{dst_host} | grep -v grep | awk '{{print $2}}' | head -1) -n iperf -s -D 2>/dev/null; sleep 1; echo 12345678 | sudo -S nsenter -t $(echo 12345678 | sudo -S ps aux | grep mininet:{src_host} | grep -v grep | awk '{{print $2}}' | head -1) -n iperf -c 10.0.0.2 -t {duration} -P 1 -f m 2>/dev/null | tail -5"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    for line in result.stdout.split('\n'):
        if 'Mbits/sec' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'Mbits/sec':
                    try:
                        return float(parts[i-1])
                    except:
                        pass
    return 0.0

def main():
    print("="*60)
    print("ONOS OF Rules: OSPF vs PPO+GNN Comparison")
    print("="*60)
    
    # Check ONOS
    devices = onos_get("devices")
    n_devices = len(devices.get("devices", []))
    print(f"ONOS Devices: {n_devices}")
    
    if n_devices < 14:
        print("ERROR: Not enough devices")
        return
    
    # Get port mappings
    print("\n--- Getting Port Mappings ---")
    port_map = get_port_mapping()
    print(f"Port mappings: {len(port_map)} entries")
    
    host_ports = get_host_ports()
    print(f"Host ports: {len(host_ports)} hosts")
    
    # OSPF Benchmark (default ONOS forwarding)
    print("\n--- OSPF Benchmark (ONOS fwd, hop-count) ---")
    ospf_results = []
    for i in range(3):
        tput = run_iperf_test("h1", "h2", 5)
        ospf_results.append(tput)
        print(f"  Run {i+1}: {tput:.1f} Mbps")
    ospf_avg = sum(ospf_results) / len(ospf_results)
    print(f"  OSPF Average: {ospf_avg:.1f} Mbps")
    
    # Install PPO+GNN rules
    print("\n--- Installing PPO+GNN Rules ---")
    try:
        rules_installed = setup_ppo_rules(port_map, host_ports)
    except Exception as e:
        print(f"Error installing rules: {e}")
        rules_installed = 0
    
    # Wait for rules to take effect
    time.sleep(3)
    
    # PPO+GNN Benchmark
    print("\n--- PPO+GNN Benchmark ---")
    ppo_results = []
    for i in range(3):
        tput = run_iperf_test("h1", "h2", 5)
        ppo_results.append(tput)
        print(f"  Run {i+1}: {tput:.1f} Mbps")
    ppo_avg = sum(ppo_results) / len(ppo_results)
    print(f"  PPO+GNN Average: {ppo_avg:.1f} Mbps")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"OSPF:    {ospf_avg:.1f} Mbps")
    print(f"PPO+GNN: {ppo_avg:.1f} Mbps")
    if ospf_avg > 0:
        improvement = ((ppo_avg - ospf_avg) / ospf_avg) * 100
        print(f"Improvement: {improvement:+.1f}%")
    
    # Save results
    results = {
        "ospp_avg": ospf_avg,
        "ppo_avg": ppo_avg,
        "improvement_pct": improvement if ospf_avg > 0 else 0,
        "ospp_results": ospf_results,
        "ppo_results": ppo_results,
        "rules_installed": rules_installed
    }
    with open("/home/ino/of_rules_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /home/ino/of_rules_comparison.json")

if __name__ == "__main__":
    main()
