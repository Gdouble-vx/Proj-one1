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
    Standard benchmark topology used in SDN, DRL, and GNN routing research (e.g., RouteNet).
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

        # Real-World NSFNET Backbone Links (21 Links)
        links = [
            (1, 2), (1, 3), (1, 8),
            (2, 3), (2, 7),
            (3, 4),
            (4, 5), (4, 6),
            (5, 6), (5, 7),
            (6, 13), (6, 14),
            (7, 8),
            (8, 9),
            (9, 10), (9, 12),
            (10, 11), (10, 13),
            (11, 12), (11, 14),
            (12, 13)
        ]

        for u, v in links:
            # Set default backbone link parameters (10 Mbps bandwidth, 2ms delay)
            self.addLink(switches[u], switches[v], bw=10, delay='2ms')


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

        # Real-World Abilene Backbone Links (15 Links)
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

        for u, v in links:
            self.addLink(switches[u], switches[v], bw=10, delay='5ms')


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
