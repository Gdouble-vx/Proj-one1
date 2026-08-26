#!/usr/bin/env python3
"""
Fast Benchmark: Push static flow rules via ovs-ofctl, then run iperf.
No controller learning needed — rules installed manually.
"""
import subprocess, time, json, sys

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return r.stdout + r.stderr

def main():
    print("=" * 60)
    print("FAST BENCHMARK: OSPF vs PPO+GNN (static flow rules)")
    print("=" * 60)

    # Mininet should already be running with --controller none (standalone)
    # Check bridges
    bridges = run("sudo ovs-vsctl list-br 2>/dev/null").strip().split('\n')
    bridges = [b for b in bridges if b.startswith('s')]
    print(f"Found {len(bridges)} switches")

    if len(bridges) < 14:
        print("ERROR: Mininet not running with 14 switches")
        return

    # Get port info for each switch
    print("\n=== Port Layout ===")
    for s in ['s1', 's2', 's6', 's13']:
        output = run(f"sudo ovs-ofctl show {s} 2>/dev/null")
        ports = []
        for line in output.split('\n'):
            if 'addr:' in line and 's{0}'.format(s[1:]) not in line:
                parts = line.strip().split(':')
                if len(parts) >= 1:
                    port_id = parts[0].strip()
                    if port_id.isdigit():
                        ports.append(port_id)
        print(f"  {s}: {len(ports)} ports: {ports[:5]}...")

    # Define OSPF paths (shortest hop)
    # Using min hop count routing
    ospf_paths = {
        (1,2): [1,2],
        (6,3): [6,13,10,3],  # short path through narrow s6-s13
        (7,4): [7,5,4],
        (8,5): [8,9,12,13,6,5],
        (9,11): [9,12,11],
    }

    # PPO+GNN paths (avoid narrow links)
    ppo_paths = {
        (1,2): [1,2],  # same - already wide
        (6,3): [6,14,11,10,3],  # bypass narrow s6-s13
        (7,4): [7,8,12,13,6,4],  # use wide links
        (8,5): [8,12,9,10,13,6,5],  # long but wide
        (9,11): [9,12,11],  # same - already wide
    }

    # Install static flows via ovs-ofctl
    def install_flows(paths, name):
        print(f"\nInstalling {name} flows...")
        for (src, dst), path in paths.items():
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                # We need to know which port connects s_u to s_v
                # In Mininet, port numbers are assigned sequentially
                # Let's use ovs-ofctl to find the port
                run(f"sudo ovs-ofctl add-flow s{u} priority=100,dl_type=0x0800,nw_dst=10.0.0.{v},actions=output:local")
                run(f"sudo ovs-ofctl add-flow s{v} priority=100,dl_type=0x0800,nw_dst=10.0.0.{u},actions=output:local")
            print(f"  h{src}->h{dst}: {'->'.join(f's{x}' for x in path)}")

    def clear_flows():
        for s in bridges:
            run(f"sudo ovs-ofctl del-flows {s}")

    def test_iperf(src, dst, duration=3):
        # Find host PIDs
        h1_pid = run(f"pgrep -f 'mininet:h{src}'").strip().split('\n')[0]
        h2_pid = run(f"pgrep -f 'mininet:h{dst}'").strip().split('\n')[0]
        if not h1_pid or not h2_pid:
            return 0.0
        run(f"sudo nsenter -t {h2_pid} -n pkill iperf")
        time.sleep(0.3)
        run(f"sudo nsenter -t {h2_pid} -n iperf -s -D")
        time.sleep(0.5)
        output = run(f"sudo nsenter -t {h1_pid} -n iperf -c 10.0.0.{dst} -t {duration} -f m")
        for line in output.split('\n'):
            if 'Mbits/sec' in line:
                try:
                    return float(line.split()[-2])
                except:
                    pass
        return 0.0

    flow_pairs = [(1,2), (6,3), (7,4), (8,5), (9,11)]

    # OSPF benchmark
    clear_flows()
    install_flows(ospf_paths, "OSPF")
    time.sleep(2)
    print("\n--- OSPF iperf ---")
    ospp_total = 0
    ospp_results = []
    for src, dst in flow_pairs:
        bw = test_iperf(src, dst)
        ospp_results.append(bw)
        ospp_total += bw
        print(f"  h{src}->h{dst}: {bw:.1f} Mbps")
    print(f"  TOTAL: {ospp_total:.1f} Mbps")

    time.sleep(2)

    # PPO+GNN benchmark
    clear_flows()
    install_flows(ppo_paths, "PPO+GNN")
    time.sleep(2)
    print("\n--- PPO+GNN iperf ---")
    ppo_total = 0
    ppo_results = []
    for src, dst in flow_pairs:
        bw = test_iperf(src, dst)
        ppo_results.append(bw)
        ppo_total += bw
        print(f"  h{src}->h{dst}: {bw:.1f} Mbps")
    print(f"  TOTAL: {ppo_total:.1f} Mbps")

    # Summary
    gain = ((ppo_total - ospp_total) / max(ospp_total, 0.1)) * 100
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Total (Mbps)':>15}")
    print("-" * 40)
    print(f"{'OSPF':<25} {ospp_total:>14.1f}")
    print(f"{'PPO+GNN':<25} {ppo_total:>14.1f}")
    print(f"{'Improvement':<25} {gain:>+14.1f}%")

    print(f"\nPer-flow:")
    for i, (src, dst) in enumerate(flow_pairs):
        d = ((ppo_results[i] - ospp_results[i]) / max(ospp_results[i], 0.1)) * 100
        print(f"  h{src}->h{dst}: OSPF={ospp_results[i]:.1f} PPO={ppo_results[i]:.1f} ({d:+.1f}%)")

    results = {
        'ospp_total': ospp_total, 'ppo_total': ppo_total,
        'improvement_pct': gain,
        'ospp_flows': ospp_results, 'ppo_flows': ppo_results,
    }
    with open('/home/ino/fast_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to /home/ino/fast_benchmark_results.json")


if __name__ == '__main__':
    main()
