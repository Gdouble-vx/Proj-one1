from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel
 
class AdvancedNeuralNetTopo(Topo):
    def build(self):
        # สร้างสวิตช์ 14 ตัว แบ่งโครงสร้างเป็น 3 เลเยอร์ (4 -> 6 -> 4) เหมือน Neural Network
        l1 = [self.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13') for i in range(1, 5)]   # s1 - s4 (Input)
        l2 = [self.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13') for i in range(5, 11)]  # s5 - s10 (Hidden/Core)
        l3 = [self.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13') for i in range(11, 15)] # s11 - s14 (Output)
 
        # สร้าง Hosts ตัวต้นทาง และ ปลายทาง
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
 
        # h1 เกาะเข้ากับสวิตช์ตัวแรก (s1) และ h2 เกาะเข้ากับสวิตช์ตัวสุดท้าย (s14)
        self.addLink(h1, l1[0])
        self.addLink(h2, l3[-1])
 
        # --- เชื่อมสายแบบ Fully Connected ระหว่างเลเยอร์ ---
 
        # 1. เชื่อม Layer 1 ไปหา Layer 2 ทุกตัว (4 x 6 = 24 ลิงก์)
        for s_in in l1:
            for s_hidden in l2:
                self.addLink(s_in, s_hidden)
 
        # 2. เชื่อม Layer 2 ไปหา Layer 3 ทุกตัว (6 x 4 = 24 ลิงก์)
        for s_hidden in l2:
            for s_out in l3:
                self.addLink(s_hidden, s_out)
 
 
def run():
    topo = AdvancedNeuralNetTopo()
 
    # เทียบเท่ากับ: --controller remote,ip=172.17.0.2,port=6653
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
 
    # เทียบเท่ากับ: --switch ovsk,protocols=OpenFlow13
    net = Mininet(
        topo=topo,
        controller=c0,
        switch=OVSKernelSwitch,
        autoSetMacs=True,
        build=True
    )
 
    net.start()
 
    # บังคับให้ทุกสวิตช์ใช้ OpenFlow13 อีกครั้งตอน runtime (กันเคสไดรเวอร์ไม่ apply ตอน build)
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set bridge {sw.name} protocols=OpenFlow13')
 
    CLI(net)
    net.stop()
 
 
if __name__ == '__main__':
    setLogLevel('info')
    run()
