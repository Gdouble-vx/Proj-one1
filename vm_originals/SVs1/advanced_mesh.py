from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

class AdvancedMeshTopo(Topo):
    def build(self):
        # 1. สร้าง Switches ทั้งหมด 6 ตัว (รองรับ OpenFlow 1.3)
        switches = {}
        for i in range(1, 7):
            switches[f's{i}'] = self.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13')

        # 2. สร้าง Hosts ต้นทาง และ ปลายทาง
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        # 3. ต่อ Host เข้าจุดสตาร์ท-จุดจบ
        self.addLink(h1, switches['s4'])  # h1 เกาะที่ s4
        self.addLink(h2, switches['s6'])  # h2 เกาะที่ s6

        # 4. เชื่อมสายลิงก์แถวบน (Horizontal Top)
        self.addLink(switches['s1'], switches['s2'])
        self.addLink(switches['s2'], switches['s3'])

        # 5. เชื่อมสายลิงก์แถวล่าง (Horizontal Bottom)
        self.addLink(switches['s4'], switches['s5'])
        self.addLink(switches['s5'], switches['s6'])

        # 6. เชื่อมสายลิงก์แนวตั้งเชื่อมแถวบน-ล่างเข้าด้วยกัน (Vertical Cross)
        self.addLink(switches['s1'], switches['s4'])
        self.addLink(switches['s2'], switches['s5'])
        self.addLink(switches['s3'], switches['s6'])

def run():
    topo = AdvancedMeshTopo()
    # ชี้ไปหา ONOS ด่านเดิมของคุณ (172.17.0.2)
    net = Mininet(topo=topo, controller=lambda name: RemoteController(name, ip='172.17.0.2', port=6653))
    net.start()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
