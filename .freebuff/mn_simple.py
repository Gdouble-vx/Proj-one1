#!/usr/bin/env python3
"""Simple Mininet NSFNET — no delay, faster startup."""
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
        for i in range(1, 15):
            s[i] = self.addSwitch(f's{i}', dpid=f'{i:016x}', protocols='OpenFlow13')
        for i in range(1, 15):
            self.addHost(f'h{i}', ip=f'10.0.0.{i}/24', mac=f'00:00:00:00:00:{i:02x}')
            self.addLink(f'h{i}', s[i], bw=100)
        for u, v, bw in [(1,2,150),(1,3,80),(1,8,100),(2,3,20),(2,7,15),(3,4,15),
                          (4,5,100),(4,6,200),(5,6,150),(5,7,150),(6,13,15),(6,14,80),
                          (7,8,100),(8,9,20),(9,10,150),(9,12,200),(10,11,100),
                          (10,13,100),(11,12,200),(11,14,15),(12,13,80)]:
            self.addLink(s[u], s[v], bw=bw)  # no delay

net = Mininet(topo=NSFNETTopo(), switch=OVSKernelSwitch, link=TCLink,
              controller=RemoteController('c0', ip='127.0.0.1', port=6653),
              autoSetMacs=True, autoStaticArp=True)
net.start()
print(f'MN_OK {len(net.switches)}sw {len(net.hosts)}h', flush=True)
time.sleep(10)
r = net.pingAll()
print(f'PING {r}', flush=True)
signal.pause()
