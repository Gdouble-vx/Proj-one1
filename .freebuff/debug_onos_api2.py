#!/usr/bin/env python3
import json, urllib.request, base64

auth = base64.b64encode(b"karaf:karaf").decode()

def get(path):
    url = f"http://192.168.10.165:8181/onos/v1/{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Basic {auth}")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())

print("=== LINKS ===")
data = get("links")
links = data.get("links", [])
print(f"{len(links)} links")
if links:
    print("Sample link:")
    print(json.dumps(links[0], indent=2))
    # Show structure
    for k, v in links[0].items():
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n=== HOSTS ===")
data = get("hosts")
hosts = data.get("hosts", [])
print(f"{len(hosts)} hosts")
if hosts:
    print("Sample host:")
    print(json.dumps(hosts[0], indent=2))
