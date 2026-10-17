"""
api/index.py
Vercel Serverless Function Handler for CardioTwin.
Exposes Member B's DSP engine and simulation APIs via standard HTTP BaseHTTPRequestHandler.
"""

import os
import sys
import json
import csv
from http.server import BaseHTTPRequestHandler
import urllib.parse

# Ensure root directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dsp import process_cardiac_recording

# In-memory transient state for serverless execution
GLOBAL_STATE = {
    "is_recording": False,
    "raw_buffer": [],
    "last_result": None,
    "has_data": False
}

class handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.endswith("/server_info") or path == "/api/server_info":
            self.send_json({
                "status": "online",
                "engine": "CardioTwin Member B DSP Engine",
                "version": "1.0.0"
            })
            return

        if path.endswith("/latest_state") or path == "/api/latest_state":
            self.send_json({
                "has_data": GLOBAL_STATE["has_data"],
                "is_recording": GLOBAL_STATE["is_recording"],
                "result": GLOBAL_STATE["last_result"]
            })
            return

        self.send_json({"message": "CardioTwin API endpoint active"}, 200)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b"{}"

        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            body = {}

        if path.endswith("/recording_control") or path == "/api/recording_control":
            action = body.get("action", "")
            if action == "START":
                GLOBAL_STATE["is_recording"] = True
                GLOBAL_STATE["raw_buffer"] = []
                self.send_json({"status": "RECORDING_STARTED"})
            elif action == "STOP":
                GLOBAL_STATE["is_recording"] = False
                
                # Check if client passed the full sample buffer directly with STOP
                client_samples = body.get("samples", [])
                t_series = body.get("t", [])
                z_series = body.get("z", [])

                if len(client_samples) > 0 and len(t_series) == 0:
                    t_series = [item["t"] for item in client_samples if "t" in item and "z" in item]
                    z_series = [item["z"] for item in client_samples if "t" in item and "z" in item]
                
                # Fallback to server raw_buffer
                if len(t_series) == 0 and len(GLOBAL_STATE["raw_buffer"]) > 50:
                    t_series = [item["t"] for item in GLOBAL_STATE["raw_buffer"] if "t" in item and "z" in item]
                    z_series = [item["z"] for item in GLOBAL_STATE["raw_buffer"] if "t" in item and "z" in item]

                if len(t_series) > 50:
                    dsp_result = process_cardiac_recording(t_series, z_series, mode="scg")
                    GLOBAL_STATE["last_result"] = dsp_result
                    GLOBAL_STATE["has_data"] = True
                    mean_hr = dsp_result.get("feature_vector", {}).get("mean_hr_bpm", "--")
                    self.send_json({"status": "PROCESSED", "mean_hr": mean_hr, "result": dsp_result})
                else:
                    self.send_json({"status": "TOO_FEW_SAMPLES", "mean_hr": "--", "samples_count": len(t_series)})
            return

        if path.endswith("/sensor_stream") or path == "/api/sensor_stream":
            samples = body.get("samples", [])
            if GLOBAL_STATE["is_recording"]:
                GLOBAL_STATE["raw_buffer"].extend(samples)
            self.send_json({"received": len(samples), "buffer_size": len(GLOBAL_STATE["raw_buffer"])})
            return

        if path.endswith("/process_recording") or path == "/api/process_recording":
            # Direct processing endpoint for Member A and client POST
            t_list = body.get("t", [])
            z_list = body.get("z", [])
            mode = body.get("mode", "scg")
            if len(t_list) > 10 and len(z_list) > 10:
                result = process_cardiac_recording(t_list, z_list, mode=mode)
                GLOBAL_STATE["last_result"] = result
                GLOBAL_STATE["has_data"] = True
                self.send_json(result)
            else:
                self.send_json({"error": "Insufficient signal samples"}, 400)
            return

        if path.endswith("/replay_test_data") or path == "/api/replay_test_data":
            csv_path = os.path.join(ROOT_DIR, "data", "raw_phyphox_test", "Raw Data.csv")
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
                GLOBAL_STATE["last_result"] = dsp_result
                GLOBAL_STATE["has_data"] = True
                self.send_json(dsp_result)
            else:
                self.send_json({"error": "Sample data not found"}, 404)
            return

        self.send_json({"error": "Endpoint not found"}, 404)
