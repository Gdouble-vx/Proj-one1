import requests

VM1_IP = "192.168.10.165"
url = f"http://{VM1_IP}:8181/onos/v1/devices"

try:
    print("กำลังทดสอบเชื่อมต่อไปยัง ONOS Controller...")
    response = requests.get(url, auth=('onos', 'rocks'), timeout=5)
    if response.status_code == 200:
        print("✅ การเชื่อมต่อสำเร็จ!")
        print(f"ข้อมูลสถานะอุปกรณ์จาก ONOS: {response.text[:200]}...") # แสดงผลลัพธ์แค่ 200 ตัวแรก
    else:
        print(f"❌ เชื่อมต่อได้แต่ ONOS ตอบกลับด้วย Status Code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"❌ เชื่อมต่อล้มเหลว: {e}")
