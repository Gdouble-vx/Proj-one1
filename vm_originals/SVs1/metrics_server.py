from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import json
import re

def get_h1_netns():
    try:
        result = subprocess.run(
            "ps ax | grep 'mininet:h1' | grep -v grep | awk '{print $1}' | head -1",
            shell=True, capture_output=True, text=True
        )
        pid = result.stdout.strip()
        return pid
    except:
        return None

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        throughput, latency, packet_loss = 100.0, 50.0, 0.0
        
        pid = get_h1_netns()
        if pid:
            try:
                # วัด latency
                ping = subprocess.run(
                    f"nsenter -t {pid} -n ping -c 3 -W 1 10.0.0.2",
                    shell=True, capture_output=True, text=True, timeout=10
                )
                if ping.returncode == 0:
                    m = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", ping.stdout)
                    if m:
                        latency = float(m.group(1))
                    lm = re.search(r"(\d+)% packet loss", ping.stdout)
                    if lm:
                        packet_loss = float(lm.group(1)) / 100.0
            except Exception as e:
                print(f"ping error: {e}")

            try:
                # วัด throughput
                iperf = subprocess.run(
                    f"nsenter -t {pid} -n iperf -c 10.0.0.1 -t 0.5 -f m",
                    shell=True, capture_output=True, text=True, timeout=15
                )
                if iperf.returncode == 0:
                    m = re.search(r"([\d.]+)\s+Mbits/sec", iperf.stdout)
                    if m:
                        throughput = float(m.group(1))
            except Exception as e:
                print(f"iperf error: {e}")

        result = json.dumps({
            "throughput": throughput,
            "latency": latency,
            "packet_loss": packet_loss
        })
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(result.encode())

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    print("Metrics Server รันที่ port 9999...")
    HTTPServer(("0.0.0.0", 9999), MetricsHandler).serve_forever()
