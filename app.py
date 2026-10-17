#!/usr/bin/env python3
"""
app.py
CardioTwin Master Server & Unified Web Application.
Runs a zero-dependency HTTP server that hosts:
  1. Desktop Unified Analytical Dashboard (/dashboard or / on desktop)
  2. Mobile Sensor Recorder PWA (/record or / on mobile)
  3. Real-Time REST APIs for streaming accelerometer ingestion and Member B's DSP engine.
"""

import os
import sys
import json
import socket
import csv
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dsp import process_cardiac_recording

PORT = 8000

# Global shared state
STATE = {
    "is_recording": False,
    "raw_buffer": [], # buffer of {t, x, y, z}
    "last_result": None,
    "has_data": False
}

def get_lan_ip():
    """Finds the local network IP of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually establish a connection; just resolves route
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class CardioServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet down terminal logging for high-rate sensor streams
        if "/api/sensor_stream" not in self.path:
            super().log_message(format, *args)

    def is_mobile_client(self):
        ua = self.headers.get("User-Agent", "").lower()
        return any(term in ua for term in ["mobile", "android", "iphone", "ipad", "ipod", "silk"])

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, file_path, content_type):
        if not os.path.exists(file_path):
            self.send_error(404, "File not found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Static Assets
        if path.startswith("/static/"):
            rel_path = path[1:] # strip leading /
            if rel_path.endswith(".json"):
                self.send_file(rel_path, "application/json")
            elif rel_path.endswith(".js"):
                self.send_file(rel_path, "application/javascript")
            elif rel_path.endswith(".css"):
                self.send_file(rel_path, "text/css")
            elif rel_path.endswith(".png"):
                self.send_file(rel_path, "image/png")
            else:
                self.send_file(rel_path, "text/plain")
            return

        # Mobile Recorder
        if path == "/record":
            self.send_file("static/mobile.html", "text/html")
            return

        # Desktop Dashboard
        if path == "/dashboard":
            self.send_file("static/dashboard.html", "text/html")
            return

        # Root: Intelligent Device Router
        if path == "/":
            if self.is_mobile_client():
                self.send_file("static/mobile.html", "text/html")
            else:
                self.send_file("static/dashboard.html", "text/html")
            return

        # API: Server Info
        if path == "/api/server_info":
            lan_ip = get_lan_ip()
            self.send_json({
                "lan_ip": lan_ip,
                "lan_url": f"http://{lan_ip}:{PORT}"
            })
            return

        # API: Latest State
        if path == "/api/latest_state":
            self.send_json({
                "has_data": STATE["has_data"],
                "is_recording": STATE["is_recording"],
                "result": STATE["last_result"]
            })
            return

        self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b"{}"

        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            body = {}

        # API: Recording Control (START / STOP)
        if path == "/api/recording_control":
            action = body.get("action", "")
            if action == "START":
                STATE["is_recording"] = True
                STATE["raw_buffer"] = []
                self.send_json({"status": "RECORDING_STARTED"})
            elif action == "STOP":
                STATE["is_recording"] = False
                # Process the accumulated buffer through Member B's DSP!
                if len(STATE["raw_buffer"]) > 50:
                    t_list = [item["t"] for item in STATE["raw_buffer"]]
                    z_list = [item["z"] for item in STATE["raw_buffer"]]
                    
                    # Run Member B DSP
                    dsp_result = process_cardiac_recording(t_list, z_list, mode="scg")
                    STATE["last_result"] = dsp_result
                    STATE["has_data"] = True
                    mean_hr = dsp_result.get("feature_vector", {}).get("mean_hr_bpm", "--")
                    self.send_json({"status": "PROCESSED", "mean_hr": mean_hr})
                else:
                    self.send_json({"status": "TOO_FEW_SAMPLES", "mean_hr": "--"})
            return

        # API: Sensor Stream (Ingesting batches from phone)
        if path == "/api/sensor_stream":
            samples = body.get("samples", [])
            if STATE["is_recording"]:
                STATE["raw_buffer"].extend(samples)
            self.send_json({"received": len(samples), "buffer_size": len(STATE["raw_buffer"])})
            return

        # API: Replay Test Data (for live laptop demo)
        if path == "/api/replay_test_data":
            csv_path = "data/raw_phyphox_test/Raw Data.csv"
            if os.path.exists(csv_path):
                timestamps = []
                z_data = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) >= 4:
                            try:
                                timestamps.append(float(row[0]))
                                z_data.append(float(row[3]))
                            except ValueError:
                                continue
                
                dsp_result = process_cardiac_recording(timestamps, z_data, mode="scg")
                STATE["last_result"] = dsp_result
                STATE["has_data"] = True
                self.send_json(dsp_result)
            else:
                self.send_json({"error": "Sample data not found"}, status=404)
            return

        self.send_error(404, "Unknown API")

def run_server():
    lan_ip = get_lan_ip()
    server_address = ('0.0.0.0', PORT)
    httpd = ThreadingHTTPServer(server_address, CardioServerHandler)
    
    print("=" * 70)
    print(" CARDIO-TWIN: UNIFIED DUAL-SCREEN PWA & ANALYTICAL DASHBOARD")
    print("=" * 70)
    print(f" Server running on port {PORT}:")
    print(f"   💻 Laptop Dashboard: http://localhost:{PORT}")
    print(f"   📱 Mobile Sensor:    http://{lan_ip}:{PORT}/record")
    print("=" * 70)
    print(" Open the laptop URL on your computer, and scan the QR code on your phone!")
    print(" Press Ctrl+C to terminate.")
    print("=" * 70)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
