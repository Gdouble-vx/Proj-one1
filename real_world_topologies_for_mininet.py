#!/usr/bin/env python3
"""
Real-World Network Topologies for Mininet & ONOS Controller
Includes:
1. NSFNET (National Science Foundation Network) - 14 Nodes, 21 Links
2. Abilene Network (Internet2 US Backbone) - 12 Nodes, 15 Links
"""

import argparse
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info


class NSFNETTopo(Topo):
    """
    NSFNET Topology (14 Switches, 21 Bidirectional Links)
    Standard benchmark topology used in SDN, DRL, and GNN routing research.
    Asymmetric link capacities: shortest-hop paths have NARROW links (bottleneck),
    while longer bypass routes use WIDE links (high bandwidth).
    OSPF (hop-count) → hits 10-15 Mbps bottlenecks
    PPO+GNN → learns to route through 100-200 Mbps bypass links
    """
    def __init__(self, **opts):
        super(NSFNETTopo, self).__init__()

        # Create 14 Switches
        switches = {}
        for i in range(1, 15):
            switches[i] = self.addSwitch(f's{i}', dpid=f'{i:016x}')

        # Create 1 Host per Switch for Traffic Generation (iperf)
        for i in range(1, 15):
            host = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24', mac=f'00:00:00:00:00:{i:02x}')
            self.addLink(host, switches[i], bw=100)  # 100 Mbps host link

        # Real-World NSFNET Backbone Links (21 Links) with ASYMMETRIC bandwidth
        # Must match network_sim.py _nsfnet_topology() exactly!
        links = [
            (1, 2), (1, 3), (1, 8),       # s1 connections
            (2, 3), (2, 7),               # s2 connections
            (3, 4),                       # s3->s4
            (4, 5), (4, 6),              # s4 connections
            (5, 6), (5, 7),              # s5 connections
            (6, 13), (6, 14),            # s6 connections
            (7, 8),                       # s7->s8
            (8, 9),                       # s8->s9
            (9, 10), (9, 12),            # s9 connections
            (10, 11), (10, 13),          # s10 connections
            (11, 12), (11, 14),          # s11 connections
            (12, 13)                     # s12->s13
        ]

        # Bandwidth (Mbps) and delay (ms) per link — matches network_sim.py exactly
        # Shortest-hop paths hit NARROW links (10-15 Mbps) → bottleneck
        # PPO+GNN can route through WIDE links (100-200 Mbps) for higher throughput
        bw_list = [
            150,  # [0] s1-s2 : wide
             80,  # [1] s1-s3 : medium bypass
            100,  # [2] s1-s8 : wide
             20,  # [3] s2-s3 : NARROW choke (shortest hop hits)
             15,  # [4] s2-s7 : NARROW alternate
             15,  # [5] s3-s4 : NARROW choke (s1->s4 shortest must use)
            100,  # [6] s4-s5 : wide backbone
            200,  # [7] s4-s6 : fastest core
            150,  # [8] s5-s6 : wide core
            150,  # [9] s5-s7 : wide bypass
             15,  # [10] s6-s13: NARROW choke
             80,  # [11] s6-s14: medium
            100,  # [12] s7-s8 : wide ring
             20,  # [13] s8-s9 : NARROW south leg
            150,  # [14] s9-s10: wide
            200,  # [15] s9-s12: fastest south
            100,  # [16] s10-s11: wide
            100,  # [17] s10-s13: wide bypass
            200,  # [18] s11-s12: fast
             15,  # [19] s11-s14: NARROW choke
             80,  # [20] s12-s13: medium
        ]

        for i, (u, v) in enumerate(links):
            bw = bw_list[i]
            # Narrow links get higher delay (simulating congestion/longer distance)
            if bw <= 15:
                delay = '4ms'   # Narrow = high delay (congested)
            elif bw <= 80:
                delay = '2ms'   # Medium = normal delay
            else:
                delay = '1ms'   # Wide = low delay (fast backbone)
            self.addLink(switches[u], switches[v], bw=bw, delay=delay)


class AbileneTopo(Topo):
    """
    Abilene Network Topology (12 Switches, 15 Bidirectional Links)
    High-speed US Internet2 academic backbone topology.
    """
    def __init__(self, **opts):
        super(AbileneTopo, self).__init__()

        # Create 12 Switches
        switches = {}
        for i in range(1, 13):
            switches[i] = self.addSwitch(f's{i}', dpid=f'{i:016x}')

        # Create Hosts
        for i in range(1, 13):
            host = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24', mac=f'00:00:00:00:00:{i:02x}')
            self.addLink(host, switches[i], bw=100)

        # Real-World Abilene Backbone Links (15 Links) — ASYMMETRIC
        # Must match network_sim.py _abilene_topology() exactly!
        links = [
            (1, 2), (1, 5),
            (2, 3), (2, 4),
            (3, 6),
            (4, 5), (4, 7),
            (5, 6),
            (6, 8),
            (7, 8), (7, 11),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12)
        ]

        bw_list = [
            150,  # s1-s2 : wide
             20,  # s1-s5 : NARROW choke
            100,  # s2-s3 : wide
             62,  # s2-s4 : medium
             15,  # s3-s6 : NARROW choke
             62,  # s4-s5 : medium
            100,  # s4-s7 : wide
            200,  # s5-s6 : fastest backbone
            100,  # s6-s8 : wide
             62,  # s7-s8 : medium
             15,  # s7-s11: NARROW choke
            100,  # s8-s9 : wide
            200,  # s9-s10: fastest backbone
            100,  # s10-s11: wide
             62,  # s11-s12: medium
        ]

        for i, (u, v) in enumerate(links):
            bw = bw_list[i]
            if bw <= 15:
                delay = '4ms'
            elif bw <= 80:
                delay = '2ms'
            else:
                delay = '1ms'
            self.addLink(switches[u], switches[v], bw=bw, delay=delay)


def run_experiment(topo_name, controller_ip, controller_port):
    setLogLevel('info')

    info(f"*** Instantiating '{topo_name}' Real-World Network Topology...\n")
    if topo_name.lower() == 'nsfnet':
        topo = NSFNETTopo()
    elif topo_name.lower() == 'abilene':
        topo = AbileneTopo()
    else:
        raise ValueError(f"Unknown topology: {topo_name}. Choose 'nsfnet' or 'abilene'.")

    info(f"*** Connecting to Remote ONOS Controller at {controller_ip}:{controller_port}...\n")
    controller = RemoteController('c0', ip=controller_ip, port=controller_port)

    net = Mininet(
        topo=topo,
        switch=OVSKernelSwitch,
        link=TCLink,
        controller=controller,
        autoSetMacs=True,
        autoStaticArp=True
    )

    info("*** Starting Network Infrastructure...\n")
    net.start()

    info("*** Testing Basic Network Connectivity (Ping All Hosts)...\n")
    net.pingAll()

    info("\n*** Network is Ready! Entering Mininet CLI. Type 'exit' or Ctrl+D to stop.\n")
    CLI(net)

    info("*** Stopping Network Infrastructure...\n")
    net.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mininet Real-World Topology Launcher for ONOS")
    parser.add_argument('--topo', type=str, default='nsfnet', choices=['nsfnet', 'abilene'],
                        help="Topology name: 'nsfnet' (14 nodes) or 'abilene' (12 nodes)")
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help="Remote ONOS Controller IP address")
    parser.add_argument('--port', type=int, default=6653,
                        help="OpenFlow port for ONOS (default: 6653)")

    args = parser.parse_args()
    run_experiment(args.topo, args.ip, args.port)
